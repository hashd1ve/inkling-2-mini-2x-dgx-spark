"""Cronometraje por componente del prefill (solo diagnóstico, PREFILL_PROF=1). Sincroniza la GPU antes y después de cada
función envuelta, así que frena el servidor: no usar para medir velocidad. Imprime una tabla acumulada cada PROF_EVERY
pasadas de prefill (lotes de >= 1024 tokens) y la reinicia."""
import importlib
import os
import sys
import time

import torch

_acc, _cnt, _on, _n, _tok, _t_wall = {}, {}, [False], [0], [0], [time.time()]
_EVERY = int(os.environ.get("PROF_EVERY", "16"))


def _wrap(owner, name, label):
    fn = getattr(owner, name)

    def w(*a, **k):
        if not _on[0]:
            return fn(*a, **k)
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        _acc[label] = _acc.get(label, 0.0) + time.perf_counter() - t
        _cnt[label] = _cnt.get(label, 0) + 1
        return r

    setattr(owner, name, w)


def _wrap_top(cls):
    fn = cls.forward

    def w(self, *a, **k):
        n = max([x.shape[0] for x in list(a) + list(k.values()) if isinstance(x, torch.Tensor) and x.dim() >= 1] or [0])
        _on[0] = n >= 1024
        if not _on[0]:
            return fn(self, *a, **k)
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(self, *a, **k)
        torch.cuda.synchronize()
        _on[0] = False
        _acc["TOTAL"] = _acc.get("TOTAL", 0.0) + time.perf_counter() - t
        _n[0] += 1
        _tok[0] += n
        if _n[0] % _EVERY == 0:
            tot = _acc["TOTAL"]
            rows = sorted(((v, kk) for kk, v in _acc.items() if kk != "TOTAL"), reverse=True)
            s = " | ".join(f"{kk} {100 * v / tot:.1f}% ({_cnt[kk] // _EVERY}x {1e3 * v / _EVERY:.0f}ms)" for v, kk in rows)
            print(f"PROF {_tok[0]} tok en {tot:.2f}s GPU ({_tok[0] / tot:.0f} tok/s), pared {time.time() - _t_wall[0]:.2f}s :: {s}",
                  file=sys.stderr, flush=True)
            _acc.clear(); _cnt.clear(); _tok[0] = 0
        if _n[0] % _EVERY == 0 or _n[0] % _EVERY == 1:
            _t_wall[0] = time.time()
        return r

    cls.forward = w


def install():
    P = "vllm.models.inkling2.nvidia."
    m = {n: importlib.import_module(P + n) for n in ("model", "attention", "sconv", "loop", "moe")}
    _wrap_top(m["loop"].Inkling2LoopModel)
    A = m["attention"].Inkling2Attention
    _wrap(A, "forward", "attn")
    _wrap(A, "_project", "attn.project")
    _wrap(A, "_attention", "attn.core")
    _wrap(m["sconv"].Inkling2ShortConv, "conv_mix", "conv_mix")
    _wrap(m["loop"].Inkling2LoopDecoderLayer, "blend", "blend")
    _wrap(m["moe"].Inkling2MoE, "forward", "moe")
    _wrap(m["moe"].Inkling2Experts, "forward", "moe.experts")
    _wrap(m["moe"].Inkling2Experts, "_gemm", "moe.gemm")
    _wrap(m["moe"], "route", "moe.route")
    from vllm.distributed.parallel_state import GroupCoordinator
    _wrap(GroupCoordinator, "all_reduce", "all_reduce")
    _wrap(GroupCoordinator, "all_gather", "all_gather")
    print("PROF instalado", file=sys.stderr, flush=True)
