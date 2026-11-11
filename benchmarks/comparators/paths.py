"""Where each comparator lives: under $COMPARATORS_HOME (default ~/scratch/comparators), as built by
install.sh. Per-tool env vars (SHAEP_BIN, ALIGNIT_BIN, PHESA_JAR, PHESA_CLASSDIR, JAVA_BIN,
COMPARATORS_OBABEL_LIB) override the prefix-derived defaults."""
from __future__ import annotations

import configparser
import hashlib
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent                      # .../Shepherd-Score-Paper
LOCK_PATH = HERE / "versions.lock"

HOME = Path(os.environ.get("COMPARATORS_HOME", os.path.expanduser("~/scratch/comparators")))

BIN_DIR, LIB_DIR, JAR_DIR = HOME / "bin", HOME / "lib", HOME / "jar"
CLASS_DIR, SRC_DIR = HOME / "classes", HOME / "src"

# The PheSA harness source; install.sh compiles it from here.
PHESA_JAVA_SRC = REPO_ROOT / "benchmarks" / "throughput" / "aligners" / "phesa_java" / "PheSAScreen.java"

# OpenBabel lives in the align_it conda env (align-it links it).
OPENBABEL_LIB = Path(os.environ.get(
    "COMPARATORS_OBABEL_LIB", os.path.expanduser("~/.conda/envs/align_it/lib")))


def _lock() -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    cp.read(LOCK_PATH)
    return cp


def shaep_bin() -> Path:
    return Path(os.environ.get("SHAEP_BIN", BIN_DIR / "shaep"))


def align_it_bin() -> Path:
    return Path(os.environ.get("ALIGNIT_BIN", BIN_DIR / "align_it"))


def phesa_jar() -> Path:
    return Path(os.environ.get("PHESA_JAR", JAR_DIR / "openchemlib-2024.4.2.jar"))


def phesa_classdir() -> Path:
    return Path(os.environ.get("PHESA_CLASSDIR", CLASS_DIR))


def java_bin() -> str:
    """Locate java: JAVA_BIN, then PATH, JAVA_HOME, then the jdk17 conda env (PheSA needs JDK 17)."""
    explicit = os.environ.get("JAVA_BIN")
    if explicit:
        return explicit
    for cand in (shutil.which("java"),
                 os.path.join(os.environ.get("JAVA_HOME", ""), "bin", "java"),
                 os.path.expanduser("~/.conda/envs/jdk17/bin/java")):
        if cand and os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return "java"                                # let it fail loudly at exec with a clear error


def phesa_classpath() -> str:
    return f"{phesa_jar()}{os.pathsep}{phesa_classdir()}"


def tool_env(tool: str) -> dict:
    """Process env with the comparator libs prepended to LD_LIBRARY_PATH (same env for every tool)."""
    parts = [str(LIB_DIR), str(OPENBABEL_LIB), os.environ.get("LD_LIBRARY_PATH", "")]
    return {**os.environ, "LD_LIBRARY_PATH": ":".join(p for p in parts if p)}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify() -> dict:
    """Manual check (`python -m comparators.paths`) of every comparator against versions.lock.

    Returns {tool: status}; raises RuntimeError if a tool is missing, the PheSAScreen class is
    older than its source, or a checksum with enforce_sha = yes does not match."""
    lock = _lock()
    artifacts = {
        "shaep":       (shaep_bin(),    "shaep"),
        "align_it":    (align_it_bin(), "align_it"),
        "openchemlib": (phesa_jar(),    "openchemlib"),
    }
    status, problems = {}, []
    for tool, (path, section) in artifacts.items():
        if not path.exists():
            status[tool] = f"MISSING ({path})"
            problems.append(f"{tool}: not found at {path} -- run benchmarks/comparators/install.sh")
            continue
        want = lock.get(section, "sha256", fallback=None)
        enforce = lock.getboolean(section, "enforce_sha", fallback=False)
        got = _sha256(path) if want else None
        if want and got != want:
            msg = f"{tool}: sha256 {got[:12]}.. != locked {want[:12]}.."
            if enforce:
                problems.append(msg + "  (ENFORCED -- refusing to run)")
                status[tool] = "SHA MISMATCH"
            else:
                status[tool] = "ok (rebuilt; sha differs from locked artifact, expected)"
        else:
            status[tool] = "ok"

    cls = phesa_classdir() / "PheSAScreen.class"
    if not cls.exists():
        status["phesa_screen"] = f"MISSING ({cls})"
        problems.append(f"phesa_screen: {cls} not found -- run install.sh")
    elif PHESA_JAVA_SRC.exists() and cls.stat().st_mtime < PHESA_JAVA_SRC.stat().st_mtime:
        status["phesa_screen"] = "STALE"
        problems.append(
            f"phesa_screen: {cls} is OLDER than {PHESA_JAVA_SRC}. The compiled harness does not "
            f"match the repo source -- re-run install.sh.")
    else:
        status["phesa_screen"] = "ok"

    status["roshambo2"] = "ok (import it inside the roshambo2 conda env)"

    if problems:
        raise RuntimeError("comparator verification FAILED:\n  " + "\n  ".join(problems))
    return status


if __name__ == "__main__":
    print(f"COMPARATORS_HOME = {HOME}")
    for k, v in verify().items():
        print(f"  {k:14} {v}")
