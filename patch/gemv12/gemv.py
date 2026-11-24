# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-M BF16 tensor-core projections for SM120/SM121."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MAX_M = 16
_BN, _BK = 32, 256


@triton.jit
def _gemv_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


# --- inkling-lab: pesos bf16 empaquetados sin pérdida en 12 bits (GEMV12=1) -------------------------------------
# Un bf16 es signo (1) + exponente (8) + mantisa (7). El exponente de casi todos los pesos cae en una ventana de 15
# valores, así que se guarda en 4 bits (códigos 1..15): plano A = 1 byte por peso (signo + mantisa) y plano B = medio
# byte (código). El código 0 marca un peso fuera de la ventana (~2 de cada 10.000): su exponente va en un byte por
# columna y tesela (256 k); si una columna tiene dos distintos en la misma tesela (raro), esos pesos van en una lista.
# El kernel reconstruye exactamente los mismos bf16 en registros y hace el mismo tl.dot que _gemv_kernel, así que la
# salida es idéntica bit a bit leyendo 1,5 bytes por peso en vez de 2. Dos detalles que lo hacen exacto:
# - Los planos se cargan como palabras de 16 bits (2 y 4 pesos consecutivos en k por palabra). Si se cargan como bytes,
#   Triton empaqueta los operandos del MMA con kWidth = 4 en vez de 2 y cambia el orden de la suma en fp32.
# - Los pesos fuera de ventana se corrigen dentro de la tesela, antes del producto, no con un término aparte.
import os

_GEMV12 = os.environ.get("GEMV12") == "1"
_GEMV12_MIN = int(os.environ.get("GEMV12_MIN_BYTES", "0"))
_packed: dict = {}
# Teselas más estrechas que las de _gemv_kernel: más bloques leyendo memoria a la vez (la salida no cambia).
_BN12 = int(os.environ.get("GEMV12_BN", "8"))
_WARPS12 = int(os.environ.get("GEMV12_WARPS", "4"))


@triton.jit
def _gemv12_kernel(
    X,
    WA,
    WB,
    EC,
    CONF,
    EO,
    EL,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SA,
    SB,
    SY,
    EB,
    NKB,
    HAS_BIAS: tl.constexpr,
    ESC: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    conf = tl.load(CONF + pid)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        t = pid * NKB + k0 // BK
        # Planos rellenados hasta un múltiplo de BN columnas y K múltiplo de BK: sin máscaras.
        pa = tl.load(WA + cols[:, None].to(tl.int64) * SA + (k0 // 2 + tl.arange(0, BK // 2))[None, :]).to(tl.uint16)
        pb = tl.load(WB + cols[:, None].to(tl.int64) * SB + (k0 // 4 + tl.arange(0, BK // 4))[None, :]).to(tl.uint16)
        a = tl.interleave(pa & 0xFF, pa >> 8)
        c = tl.interleave(
            tl.interleave(pb & 15, (pb >> 8) & 15), tl.interleave((pb >> 4) & 15, pb >> 12)
        )
        expo = c + EB
        if ESC:
            # Código 0 = fuera de la ventana: su exponente es el byte de su columna en esta tesela.
            ec = tl.load(EC + t.to(tl.int64) * BN + tl.arange(0, BN)).to(tl.uint16)
            expo = tl.where(c == 0, ec[:, None], expo)
        bits = (((a & 0x80) << 8) | (expo << 7) | (a & 0x7F)).to(tl.uint16)
        if ESC:
            if (conf >> (k0 // BK)) & 1:
                # Raro: una columna con dos exponentes de escape distintos en la misma tesela. Sus pesos van aparte.
                pos = tl.arange(0, BN)[:, None] * BK + tl.arange(0, BK)[None, :]
                eo = tl.load(EO + t)
                for j in range(eo & 0xFF):
                    v = tl.load(EL + (eo >> 8) + j)
                    bits = tl.where(pos == (v >> 16), (v & 0xFFFF).to(tl.uint16), bits)
        w = tl.trans(bits.to(tl.bfloat16, bitcast=True))
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def _s16(v: torch.Tensor) -> torch.Tensor:
    return ((v ^ 0x8000) - 0x8000).to(torch.int16)


def pack12(weight: torch.Tensor):
    """(plano A [Np, K/2] int16, plano B [Np, K/4] int16, exponentes de escape [teselas, BN] uint8, máscara de teselas
    con colisión por bloque, inicio << 8 | cuántos por tesela, lista posición << 16 | bf16, base - 1, teselas en k), o
    None si no encaja."""
    n, k = weight.shape
    if k % _BK or k > 31 * _BK or n == 0:
        return None
    dev = weight.device
    bits = weight.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exp = (bits >> 7) & 0xFF
    hist = torch.bincount(exp.flatten(), minlength=256).cumsum(0)
    win = hist[14:] - torch.cat([hist.new_zeros(1), hist[:-15]])
    base = int(win.argmax())
    code = exp - base + 1
    esc = (code < 1) | (code > 15)
    code[esc] = 0
    np_ = (n + _BN12 - 1) // _BN12 * _BN12
    nkb, npid = k // _BK, np_ // _BN12
    a = torch.zeros((np_, k), dtype=torch.int32, device=dev)
    a[:n] = ((bits >> 8) & 0x80) | (bits & 0x7F)
    plane_a = _s16(a[:, 0::2] | (a[:, 1::2] << 8)).contiguous()
    del a
    cc = torch.zeros((np_, k), dtype=torch.int32, device=dev)
    cc[:n] = code
    plane_b = _s16(cc[:, 0::4] | (cc[:, 1::4] << 4) | (cc[:, 2::4] << 8) | (cc[:, 3::4] << 12)).contiguous()
    del cc, code
    # Tramo = una columna dentro de una tesela (256 k). Un exponente de escape por tramo; si un tramo tiene dos
    # distintos hay colisión y todos sus escapes van a la lista de su tesela.
    r, c = esc.nonzero(as_tuple=True)
    e = exp[r, c]
    run = r * nkb + c // _BK
    lo = torch.full((np_ * nkb,), 255, dtype=torch.int32, device=dev).scatter_reduce(0, run, e, "amin")
    hi = torch.zeros(np_ * nkb, dtype=torch.int32, device=dev).scatter_reduce(0, run, e, "amax")
    ec = hi.view(npid, _BN12, nkb).transpose(1, 2).contiguous().to(torch.uint8)  # [bloque, tesela en k, columna]
    bad = (lo[run] != hi[run])
    r, c = r[bad], c[bad]
    tile = (r // _BN12) * nkb + c // _BK
    order = torch.argsort(tile, stable=True)
    r, c, tile = r[order], c[order], tile[order]
    counts = torch.bincount(tile, minlength=npid * nkb)
    if int(counts.max()) > 255 or r.numel() >= 1 << 23:
        return None
    eo = (((counts.cumsum(0) - counts) << 8) | counts).to(torch.int32)
    el = ((((r % _BN12) * _BK + c % _BK) << 16) | bits[r, c]).to(torch.int32)
    if el.numel() == 0:
        el = el.new_zeros(1)
    conf = ((counts.view(npid, nkb) > 0).to(torch.int32) << torch.arange(nkb, device=dev, dtype=torch.int32)).sum(1)
    return plane_a, plane_b, ec, conf.to(torch.int32), eo, el.contiguous(), base - 1, nkb


def _get_packed(weight: torch.Tensor):
    key = (weight.data_ptr(), weight.shape, weight.stride(0))
    hit = _packed.get(key)
    if hit is None:
        # Guarda también el peso: mientras viva, su dirección no se reutiliza para otro tensor.
        hit = _packed[key] = (pack12(weight), weight)
        if os.environ.get("GEMV12_DEBUG") == "1":
            print(f"gemv12: {tuple(weight.shape)} {'empaquetado' if hit[0] else 'sin empaquetar'} "
                  f"({len(_packed)} en caché)", flush=True)
    return hit[0]


def sm12x_bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    assert m <= MAX_M
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    packed = None
    if _GEMV12 and m and n * k * 2 >= _GEMV12_MIN:
        packed = _get_packed(weight)
    if packed is not None:
        plane_a, plane_b, ec, conf, eo, el, eb, nkb = packed
        _gemv12_kernel[(triton.cdiv(n, _BN12),)](
            x2,
            plane_a,
            plane_b,
            ec,
            conf,
            eo,
            el,
            bias,
            out,
            m,
            n,
            k,
            x2.stride(0),
            plane_a.stride(0),
            plane_b.stride(0),
            out.stride(0),
            eb,
            nkb,
            HAS_BIAS=bias is not None,
            ESC=os.environ.get("GEMV12_NOESC") != "1",  # solo para medir: sin escapes no es exacto
            BN=_BN12,
            BK=_BK,
            MP=16,
            num_warps=_WARPS12,
        )
        return out.view(*lead, n)
    if m:
        _gemv_kernel[(triton.cdiv(n, _BN),)](
            x2,
            weight,
            bias,
            out,
            m,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            HAS_BIAS=bias is not None,
            BN=_BN,
            BK=_BK,
            MP=16,
            num_warps=4,
        )
    return out.view(*lead, n)


def use_sm12x_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    return (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim >= 2
        and weight.ndim == 2
        and x.shape[-1] == weight.shape[1]
        and x.stride(-1) == weight.stride(-1) == 1
        and 0 < x.numel() // x.shape[-1] <= MAX_M
        and (
            bias is None
            or (
                bias.dtype == torch.bfloat16
                and bias.device == x.device
                and bias.stride(0) == 1
            )
        )
        and current_platform.is_device_capability_family(120)
    )
