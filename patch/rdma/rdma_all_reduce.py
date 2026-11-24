"""All-reduce de TP=2 entre dos DGX Spark por RDMA directo (RoCE), sin NCCL, para tensores bf16 pequeños.

NCCL sin GPUDirect tarda ~48 µs por all-reduce pequeño entre Sparks y la decodificación hace ~100 por paso. Aquí un solo
kernel de GPU copia sus filas a memoria anclada del host (en el GB10 es la misma memoria que usa la GPU), un hilo de CPU
en espera activa las manda con un RDMA WRITE a la memoria anclada del otro nodo, y el mismo kernel espera a verlas llegar
y suma. La suma es la de NCCL: cada elemento en fp32 y redondeado una vez a bf16 (con dos rangos el orden no importa).
Las dos partes ejecutan la misma secuencia de all-reduce, así que un contador en la GPU hace de número de secuencia.
"""
import ctypes
import os
import subprocess

import cupy
import numpy as np
import torch
import torch.distributed as dist

_HDR = 64
_NSLOTS = 4
_MAX_BLOCKS = 32
_SRC = r"""
typedef unsigned long long u64;
__device__ __forceinline__ float bf2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ u64 sum4(u64 a, u64 b) {
    u64 r = 0;
    for (int k = 0; k < 64; k += 16)
        r |= (u64)f2bf(bf2f((unsigned short)(a >> k)) + bf2f((unsigned short)(b >> k))) << k;
    return r;
}
// n8 palabras de 8 bytes repartidas en bloques de chunk palabras. state: [secuencia, copiados, terminados].
// Todos los bloques esperan a la vez, así que tienen que caber residentes en la GPU (como mucho MAX_BLOCKS).
// row8 = 0: all-reduce (suma de 4 bf16 por palabra). row8 > 0: all-gather por filas de row8 palabras; la salida tiene
// filas de 2 * row8 con la parte del rango 0 delante.
extern "C" __global__ void xchg(u64 x_, u64 out_, u64 send_, u64 recv_, u64 state_, int n8, int chunk, u64 slot_bytes,
                                int row8, int rank) {
    const u64 *x = (const u64 *)x_;
    u64 *out = (u64 *)out_;
    int *state = (int *)state_;
    int tid = threadIdx.x, nt = blockDim.x, nb = gridDim.x;
    int lo = blockIdx.x * chunk, hi = min(n8, lo + chunk);
    // Solo el último bloque en acabar avanza la secuencia, así que todos leen la misma.
    u64 seq = (u64)(unsigned)state[0] + 1, off = (seq % NSLOTS) * slot_bytes;
    volatile u64 *sh = (volatile u64 *)(send_ + off);
    volatile u64 *rh = (volatile u64 *)(recv_ + off);
    u64 *sp = (u64 *)(send_ + off + HDR);
    const u64 *rp = (const u64 *)(recv_ + off + HDR);
    for (int i = lo + tid; i < hi; i += nt) sp[i] = x[i];
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        if (atomicAdd(&state[1], 1) + 1 == nb) {
            sh[1] = (u64)n8 * 8;
            __threadfence_system();
            sh[0] = seq;
            __threadfence_system();
        }
        while (rh[0] != seq) {}
    }
    __syncthreads();
    if (row8 == 0) {
        for (int i = lo + tid; i < hi; i += nt) out[i] = sum4(x[i], rp[i]);
    } else {
        for (int i = lo + tid; i < hi; i += nt) {
            u64 base = (u64)(i / row8) * 2 * row8 + i % row8;
            out[base + (u64)rank * row8] = x[i];
            out[base + (u64)(1 - rank) * row8] = rp[i];
        }
    }
    __syncthreads();
    if (tid == 0 && atomicAdd(&state[2], 1) + 1 == nb) {
        state[1] = 0, state[2] = 0;
        __threadfence();
        state[0] = (int)seq;
    }
}
"""


def _build_lib() -> ctypes.CDLL:
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rdma_xchg.c")
    so = f"/tmp/rdma_xchg_{os.getpid()}.so"
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, src, "-libverbs", "-lpthread"])
    return ctypes.CDLL(so)


class RdmaAllReduce:
    def __init__(self, cpu_group, device: torch.device, max_bytes: int = 128 * 2304 * 2,
                 max_gather_bytes: int | None = None) -> None:
        assert cpu_group.size() == 2
        self.rank = dist.get_rank(cpu_group)
        self.max_reduce_bytes = max_bytes
        self.max_gather_bytes = int(os.environ.get("RDMA_AG_MAX_BYTES", 1 << 20)) if max_gather_bytes is None else max_gather_bytes
        self.slot = slot = (_HDR + max(max_bytes, self.max_gather_bytes) + 4095) // 4096 * 4096
        total = 2 * _NSLOTS * slot
        from cuda.bindings import driver as cuda

        err, self.host_ptr = cuda.cuMemHostAlloc(total, cuda.CU_MEMHOSTALLOC_PORTABLE | cuda.CU_MEMHOSTALLOC_DEVICEMAP)
        assert int(err) == 0, err
        err, dev_ptr = cuda.cuMemHostGetDevicePointer(self.host_ptr, 0)
        assert int(err) == 0, err
        self.send_ptr = np.uint64(int(dev_ptr))
        self.recv_ptr = np.uint64(int(dev_ptr) + _NSLOTS * slot)
        self.lib = _build_lib()
        self.lib.xc_sent.restype = ctypes.c_uint64
        info = (ctypes.c_uint64 * 6)()
        dev_name = os.environ.get("RDMA_AR_DEV", "rocep1s0f1").encode()
        rc = self.lib.xc_init(dev_name, int(os.environ.get("RDMA_AR_GID", "5")), ctypes.c_void_p(int(self.host_ptr)),
                              ctypes.c_size_t(slot), _NSLOTS, info)
        assert rc == 0, f"xc_init {rc}"
        mine = torch.tensor(np.array(list(info), dtype=np.uint64).view(np.int64))
        both = [torch.zeros_like(mine) for _ in range(2)]
        dist.all_gather(both, mine, group=cpu_group)
        other = (ctypes.c_uint64 * 6)(*both[1 - self.rank].numpy().view(np.uint64).tolist())
        rc = self.lib.xc_connect(other, int(os.environ.get("RDMA_AR_CPU", "-1")))
        assert rc == 0, f"xc_connect {rc}"
        self.state = torch.zeros(3, dtype=torch.int32, device=device)
        self.kernel = cupy.RawKernel(_SRC, "xchg", options=(f"-DNSLOTS={_NSLOTS}", f"-DHDR={_HDR}"))
        self.kernel.compile()
        dist.barrier(group=cpu_group)

    def should_use(self, x: torch.Tensor) -> bool:
        n = x.numel()
        return x.dtype == torch.bfloat16 and x.is_contiguous() and 0 < n * 2 <= self.max_reduce_bytes and n % 4 == 0

    def should_gather(self, x: torch.Tensor, dim: int) -> bool:
        """all-gather por la última dimensión de un tensor 2D contiguo cuyas filas son múltiplos de 8 bytes."""
        if x.dim() != 2 or dim != 1 or not x.is_contiguous() or x.numel() == 0:
            return False
        row_bytes = x.shape[1] * x.element_size()
        return row_bytes % 8 == 0 and x.shape[0] * row_bytes <= self.max_gather_bytes

    def _launch(self, x: torch.Tensor, out: torch.Tensor, n8: int, row8: int) -> None:
        blocks = min(_MAX_BLOCKS, (n8 + 1023) // 1024)
        chunk = (n8 + blocks - 1) // blocks
        with cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            self.kernel((blocks,), (256 if chunk <= 4096 else 1024,),
                        (np.uint64(x.data_ptr()), np.uint64(out.data_ptr()), self.send_ptr, self.recv_ptr,
                         np.uint64(self.state.data_ptr()), np.int32(n8), np.int32(chunk), np.uint64(self.slot),
                         np.int32(row8), np.int32(self.rank)))

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self._launch(x, out, x.numel() // 4, 0)
        return out

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_empty((x.shape[0], 2 * x.shape[1]))
        row8 = x.shape[1] * x.element_size() // 8
        self._launch(x, out, x.shape[0] * row8, row8)
        return out
