"""
yapsnap.cpuopt — physical-core detection for sizing the inference thread pool.

The goal: figure out how many threads yapsnap should hand to sherpa-onnx on
whatever machine it happens to run on, with a safe fallback when the platform
doesn't expose its topology.

Why this exists
---------------
sherpa-onnx (ONNX Runtime under the hood) spins up an intra-op thread pool for
the INT8 GEMMs in the Zipformer transducer. Using ``os.cpu_count()`` — the
*logical* count — oversubscribes on SMT machines: two GEMM threads on sibling
hyperthreads just fight over the same execution ports and L1. Benchmarking
showed the total thread budget should track *physical* cores, not logical ones;
exceeding the physical count consistently made decoding slower.

This module only detects that physical-core number. yapsnap decides how to
split it across decode workers (see transcribe_sentences). Nothing here is
required for correctness — it only changes speed, and degrades to a
conservative SMT-aware guess if topology can't be read.

Public API
----------
    plan = CpuPlan.detect()    # autodetect once at startup
    plan.num_threads           # -> int, total thread budget for sherpa-onnx
    get_plan().num_threads     # process-wide cached convenience

The env var YAPSNAP_THREADS overrides the thread count (0/empty = autodetect).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Topology detection
# ---------------------------------------------------------------------------

def _env_int(name: str) -> Optional[int]:
    """Parse an int env var; return None when unset, empty, or unparseable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _physical_cores_linux() -> Optional[int]:
    """Physical core count by walking /sys topology, grouping logical CPUs by
    their (physical_package_id, core_id). Returns None if /sys is unavailable.
    """
    base = "/sys/devices/system/cpu"
    if not os.path.isdir(base):
        return None

    groups: set[tuple[int, int]] = set()
    try:
        cpu_dirs = sorted(
            d for d in os.listdir(base)
            if d.startswith("cpu") and d[3:].isdigit()
        )
    except OSError:
        return None

    for d in cpu_dirs:
        topo = os.path.join(base, d, "topology")
        try:
            with open(os.path.join(topo, "core_id")) as f:
                core_id = int(f.read().strip())
            with open(os.path.join(topo, "physical_package_id")) as f:
                pkg_id = int(f.read().strip())
        except (OSError, ValueError):
            continue
        groups.add((pkg_id, core_id))

    return len(groups) or None


def _physical_cores_generic() -> Optional[int]:
    """Physical core count via psutil if available, else None.

    psutil is not a yapsnap dependency, so this is purely opportunistic — it
    just gives macOS/Windows a better number when the user happens to have it.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        n = psutil.cpu_count(logical=False)
        return int(n) if n else None
    except Exception:
        return None


def _physical_cores_macos() -> Optional[int]:
    """Physical core count on macOS via sysctl (no psutil needed)."""
    if sys.platform != "darwin":
        return None
    # hw.perflevel0.physicalcpu = performance cores on Apple Silicon; that is
    # exactly what we want to size the pool to. Fall back to hw.physicalcpu.
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            import subprocess
            out = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                val = int(out.stdout.strip())
                if val > 0:
                    return val
        except Exception:
            continue
    return None


def _logical_count() -> int:
    """Logical CPU count, never less than 1."""
    return max(1, os.cpu_count() or 1)


def _affinity_count() -> Optional[int]:
    """How many CPUs this process is actually *allowed* to run on.

    On Linux a cgroup / taskset / container may restrict us below the physical
    count; honouring that avoids oversubscription. Returns None when the
    platform has no affinity API.
    """
    getaff = getattr(os, "sched_getaffinity", None)
    if getaff is None:
        return None
    try:
        return len(getaff(0))
    except OSError:
        return None


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass
class CpuPlan:
    """An autodetected CPU tuning plan. Build it once with ``CpuPlan.detect()``.

    Callers normally touch only ``num_threads``.
    """
    logical: int
    physical: Optional[int]
    allowed: Optional[int]
    num_threads: int
    source: str = "fallback"        # how `physical` was determined

    @classmethod
    def detect(cls) -> "CpuPlan":
        """Inspect the current machine and return a tuning plan.

        Order of preference for the physical-core count:
          1. Linux /sys topology
          2. macOS sysctl         (performance cores on Apple Silicon)
          3. psutil, if installed (covers Windows and odd platforms)
          4. logical // 2 heuristic when SMT is likely, else logical
        Then it is clamped to the process's allowed-CPU set and to >= 1.

        YAPSNAP_THREADS overrides the final count. Detection never raises —
        worst case it returns a conservative SMT-aware guess.
        """
        logical = _logical_count()
        allowed = _affinity_count()

        physical: Optional[int] = None
        source = "fallback"

        # 1. Linux /sys.
        lin_n = _physical_cores_linux()
        if lin_n:
            physical, source = lin_n, "linux-sysfs"

        # 2. macOS sysctl.
        if physical is None:
            mac_n = _physical_cores_macos()
            if mac_n:
                physical, source = mac_n, "macos-sysctl"

        # 3. psutil (opportunistic; mainly Windows).
        if physical is None:
            ps_n = _physical_cores_generic()
            if ps_n:
                physical, source = ps_n, "psutil"

        # 4. Heuristic. Most laptop CPUs with an even logical count >= 4 have
        #    2-way SMT; halving is the safe guess. Odd or tiny counts: trust
        #    logical as-is.
        if physical is None:
            if logical >= 4 and logical % 2 == 0:
                physical, source = logical // 2, "heuristic-smt"
            else:
                physical, source = logical, "heuristic-logical"

        # Base thread budget: one per physical core.
        threads = physical or logical

        # Never oversubscribe a restricted affinity set (cgroup / taskset).
        if allowed is not None:
            threads = min(threads, allowed)

        # A pool of 1 is fine on a 2-core ultrabook; just keep >= 1.
        threads = max(1, threads)

        # Env override wins outright (0 or empty = keep autodetected value).
        override = _env_int("YAPSNAP_THREADS")
        if override is not None and override > 0:
            threads = override

        return cls(
            logical=logical,
            physical=physical,
            allowed=allowed,
            num_threads=threads,
            source=source,
        )


# ---------------------------------------------------------------------------
# Module-level convenience: detect once, reuse everywhere.
# ---------------------------------------------------------------------------

_PLAN: Optional[CpuPlan] = None


def get_plan() -> CpuPlan:
    """Return the process-wide CpuPlan, detecting it on first call."""
    global _PLAN
    if _PLAN is None:
        _PLAN = CpuPlan.detect()
    return _PLAN
