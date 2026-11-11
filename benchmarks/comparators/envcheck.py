"""Preflight: CPU timing needs numba <= 0.59 (SVML) and FSS_CPU_SOA=1, or fss is silently understated;
GPU timing needs a CUDA torch. Run: python -m comparators.envcheck [numba|triton] (exit 1 if unfit; no
argument = both must be fit). From Python: envcheck.check("numba"); cell["env"] = envcheck.fingerprint()."""
from __future__ import annotations

import os
import sys


def fingerprint() -> dict:
    """Versions and flags recorded into every result cell."""
    fp = {"python": sys.version.split()[0], "using_svml": None, "numba": None,
          "llvmlite": None, "numpy": None, "torch": None, "cuda": None,
          "fss_cpu_soa": os.environ.get("FSS_CPU_SOA"),
          "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS")}
    try:
        import numba
        from numba.core import config as ncfg
        fp["numba"] = numba.__version__
        fp["using_svml"] = bool(ncfg.USING_SVML)
    except Exception:
        pass
    for mod, key in (("llvmlite", "llvmlite"), ("numpy", "numpy")):
        try:
            fp[key] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        fp["torch"] = torch.__version__
        fp["cuda"] = torch.cuda.is_available()
    except Exception:
        pass
    return fp


def problems(backend: str) -> list[str]:
    """What is wrong with this env for ``backend``; empty list = fit to benchmark."""
    fp, bad = fingerprint(), []

    if backend in ("numba", "cpu"):
        if fp["numba"] is None:
            bad.append("numba is not installed -- the CPU backend cannot run.")
        elif not fp["using_svml"]:
            bad.append(
                f"numba {fp['numba']} reports USING_SVML=False. fss's CPU kernels will fall back "
                f"to scalar loops and run up to 13.8x SLOWER (surf), 2.1x (vol) -- silently, with "
                f"no error and no wrong answer. Any CPU number measured here UNDERSTATES fss.\n"
                f"      Fix: use the pinned env (numba==0.59.1 + llvmlite==0.42.0 + numpy==1.26.4 "
                f"+ intel-cmplr-lib-rt).  See benchmarks/comparators/envs/fss.yml.\n"
                f"      NB installing libsvml.so is NOT enough -- the numba VERSION is what gates it.")
        if os.environ.get("FSS_CPU_SOA") != "1":
            bad.append("FSS_CPU_SOA=1 is not set -- the SoA CPU kernels (the ones SVML vectorises) "
                       "are disabled. export FSS_CPU_SOA=1")

    if backend in ("triton", "cuda", "gpu"):
        if fp["torch"] is None:
            bad.append("torch is not installed -- the GPU backend cannot run.")
        elif not fp["cuda"]:
            bad.append("torch.cuda.is_available() is False -- no usable GPU.")

    return bad


def check(backend: str) -> dict:
    """Raise unless this env can benchmark ``backend`` (FSS_ALLOW_BAD_ENV=1 downgrades to a warning);
    returns the fingerprint."""
    bad = problems(backend)
    if bad and os.environ.get("FSS_ALLOW_BAD_ENV") == "1":
        for b in bad:
            print(f"  !! ENV WARNING (FSS_ALLOW_BAD_ENV=1): {b}", file=sys.stderr, flush=True)
    elif bad:
        raise RuntimeError(
            "REFUSING TO BENCHMARK -- this environment would produce misleading numbers:\n  - "
            + "\n  - ".join(bad)
            + "\n\nOverride with FSS_ALLOW_BAD_ENV=1 only if you intend to publish a "
              "DISCLOSED slow-path measurement.")
    return fingerprint()


if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or ["numba", "triton"]

    fp = fingerprint()
    print("comparator/fss benchmark environment")
    for k, v in fp.items():
        print(f"    {k:18} {v}")
    print()
    fatal = False
    for backend in want:
        bad = problems(backend)
        if bad:
            print(f"  {backend:8} UNFIT:")
            for b in bad:
                print(f"      - {b}")
            fatal = True
        else:
            print(f"  {backend:8} ok")
    sys.exit(1 if fatal else 0)
