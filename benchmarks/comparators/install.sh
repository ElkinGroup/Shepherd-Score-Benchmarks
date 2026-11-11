#!/usr/bin/env bash
# Build every comparator into one prefix ($COMPARATORS_HOME, default ~/scratch/comparators) from the
# versions pinned in versions.lock. Idempotent; checksums are verified where they are meaningful.
#
#   bash benchmarks/comparators/install.sh all           # conda envs + every tool (blank machine)
#   bash benchmarks/comparators/install.sh phesa         # one tool
#   COMPARATORS_HOME=/somewhere bash ... install.sh # override the prefix
#
# ShaEP is binary-only and its upstream site is gone; the 1.4.0 zip is fetched from the Wayback
# Machine capture pinned below and both the zip and the binary are checksum-enforced.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PREFIX="${COMPARATORS_HOME:-$HOME/scratch/comparators}"

# OpenBabel + the C++ toolchain align-it links against.
OBABEL_ENV="${COMPARATORS_OBABEL_ENV:-$HOME/.conda/envs/align_it}"

ALIGNIT_REPO="https://github.com/OliverBScott/align-it.git"
ALIGNIT_COMMIT="fa734f2795cc1250004cbe199c65785af30ce4c8"
OCL_URL="https://repo1.maven.org/maven2/com/actelion/research/openchemlib/2024.4.2/openchemlib-2024.4.2.jar"
OCL_SHA="a949032aff9648317c5ceaad6b63b6485f4960ded850f764ec0edeb17ce0a9b8"
SHAEP_URL="https://web.archive.org/web/20241228132449id_/http://users.abo.fi/mivainio/shaep/Linux_64bit-ShaEP.zip"
SHAEP_ZIP_SHA="eeb77a0bccd027d2ea78e62bedb8fbf14e7e6902a3640c1f5365390192072a0a"
SHAEP_SHA="6f91f3289a79fbbe79f8a886d7cf766ceafe56786616ba96ebd9f9620a0e23dc"
ESPSIM_REPO="https://github.com/hesther/espsim"
ESPSIM_COMMIT="582b39e514f5035c08965aa07bde0a123e7bc914"

say()  { printf "\n\033[1m==> %s\033[0m\n" "$*"; }
ok()   { printf "    \033[32mok\033[0m  %s\n" "$*"; }
die()  { printf "    \033[31mFAIL\033[0m %s\n" "$*" >&2; exit 1; }
sha()  { sha256sum "$1" | cut -d' ' -f1; }

mkdir -p "$PREFIX"/{bin,lib,jar,classes,src}

# ---------------------------------------------------------------------------- align-it (Pharao)
install_align_it() {
  say "align-it (Pharao) 1.0.4  @ ${ALIGNIT_COMMIT:0:7}"
  [ -d "$OBABEL_ENV" ] || die "OpenBabel env not found: $OBABEL_ENV (set COMPARATORS_OBABEL_ENV)"
  local src="$PREFIX/src/align-it"
  if [ ! -d "$src/.git" ]; then git clone -q "$ALIGNIT_REPO" "$src"; fi
  git -C "$src" fetch -q --all && git -C "$src" checkout -q "$ALIGNIT_COMMIT"
  rm -rf "$src/build" && mkdir -p "$src/build"
  ( cd "$src/build"
    cmake -DCMAKE_INSTALL_PREFIX="$PREFIX" \
          -DCMAKE_CXX_COMPILER="$OBABEL_ENV/bin/x86_64-conda-linux-gnu-c++" \
          -DOPENBABEL3_INCLUDE_DIR="$OBABEL_ENV/include/openbabel3" \
          -DOPENBABEL3_LIBRARIES="$OBABEL_ENV/lib/libopenbabel.so" .. >/dev/null
    make -j"$(nproc)" >/dev/null && make install >/dev/null )
  [ -x "$PREFIX/bin/align_it" ] || die "align_it not produced"
  # align-it ships its own .so; make sure it landed in the shared lib dir.
  find "$src/build" -name 'libalignit*' -exec cp -f {} "$PREFIX/lib/" \; 2>/dev/null || true
  ok "built  ($(sha "$PREFIX/bin/align_it" | cut -c1-12)..)"

  # Check the default feature set with a real 1-vs-1 run (the "Hybrids:" banner is not printed by
  # --help). A build reporting "Hybrids: no" loses 2 of the 8 pharmacophore feature types.
  local tmp; tmp="$(mktemp -d)"
  printf 'C\n  Mrv  \n\n  1  0  0  0  0  0            999 V2000\n    0.0000    0.0000    0.0000 C   0  0\nM  END\n$$$$\n' > "$tmp/m.sdf"
  local banner
  banner=$(LD_LIBRARY_PATH="$PREFIX/lib:$OBABEL_ENV/lib:${LD_LIBRARY_PATH:-}" \
           "$PREFIX/bin/align_it" --reference "$tmp/m.sdf" --dbase "$tmp/m.sdf" \
           --scores "$tmp/s.tab" 2>&1 | grep -i "Hybrids" || true)
  rm -rf "$tmp"
  if [ -z "$banner" ]; then
    die "align_it printed no 'Hybrids:' banner -- cannot verify the feature set. Do not benchmark
    until this is understood (a build with hybrids silently off loses 2 of 8 feature types)."
  fi
  printf "    align_it default feature set: %s\n" "$banner"
  case "$banner" in
    *[Nn][Oo]*) die "align_it reports 'Hybrids: no' BY DEFAULT -- 2 of 8 pharmacophore feature
    types are disabled. Do not benchmark with this build." ;;
    *) ok "hybrid feature types ON by default (HYBH + HYBL present)" ;;
  esac
}

# ---------------------------------------------------------------------------- PheSA (jar + harness)
install_phesa() {
  say "PheSA: openchemlib 2024.4.2 jar + PheSAScreen (compiled FROM THE REPO)"
  local jar="$PREFIX/jar/openchemlib-2024.4.2.jar"
  if [ ! -f "$jar" ] || [ "$(sha "$jar")" != "$OCL_SHA" ]; then
    curl -fsSL -o "$jar" "$OCL_URL"
  fi
  [ "$(sha "$jar")" = "$OCL_SHA" ] || die "openchemlib jar sha256 mismatch -- refusing to continue"
  ok "jar verified against Maven Central ($(echo "$OCL_SHA" | cut -c1-12)..)"

  local jsrc="$REPO_ROOT/benchmarks/throughput/aligners/phesa_java/PheSAScreen.java"
  [ -f "$jsrc" ] || die "PheSAScreen.java not found in the repo at $jsrc"

  # `module load openjdk` does not work in a non-interactive shell; search PATH, JAVA_HOME, then
  # the jdk17 conda env.
  local JAVAC=""
  for c in "${JAVAC:-}" "$(command -v javac 2>/dev/null)" \
           "${JAVA_HOME:-}/bin/javac" "$HOME/.conda/envs/jdk17/bin/javac"; do
    [ -n "$c" ] && [ -x "$c" ] && { JAVAC="$c"; break; }
  done
  [ -n "$JAVAC" ] || die "javac not found. Tried \$PATH, \$JAVA_HOME/bin, ~/.conda/envs/jdk17/bin.
    Set JAVAC=/path/to/javac, or 'module load openjdk' in an INTERACTIVE shell first."
  printf "    javac: %s\n" "$JAVAC"
  "$JAVAC" -cp "$jar" -d "$PREFIX/classes" "$jsrc"
  [ -f "$PREFIX/classes/PheSAScreen.class" ] || die "PheSAScreen.class not produced"
  ok "PheSAScreen compiled from $(realpath --relative-to="$REPO_ROOT" "$jsrc")"
}

# ---------------------------------------------------------------------------- ShaEP (Wayback)
install_shaep() {
  say "ShaEP 1.4.0  (from the Wayback Machine -- upstream is 404 and there is no source)"
  local dst="$PREFIX/bin/shaep"
  if [ -f "$dst" ] && [ "$(sha "$dst")" = "$SHAEP_SHA" ]; then
    ok "already in the prefix, checksum matches"
    "$dst" --version 2>&1 | grep -i version | head -1 | sed 's/^/    /'; return
  fi
  # SHAEP_SOURCE forces a local copy of the binary (not reproducible; do not publish numbers from it).
  if [ -n "${SHAEP_SOURCE:-}" ] && [ -f "$SHAEP_SOURCE" ]; then
    printf "    \033[33mWARN\033[0m using SHAEP_SOURCE=%s -- NOT the reproducible 1.4.0\n" "$SHAEP_SOURCE"
    install -m 0755 "$SHAEP_SOURCE" "$dst"
  else
    local tmp; tmp="$(mktemp -d)"
    curl -fsSL --retry 3 --max-time 300 -o "$tmp/shaep.zip" "$SHAEP_URL" \
      || die "could not fetch ShaEP from the Wayback Machine ($SHAEP_URL).
    ShaEP's upstream site is dead (404), it is binary-only, and its source repo is private -- the
    Wayback capture is the ONLY reproducible source. If it is unreachable, retry later or restore
    from a preserved copy via SHAEP_SOURCE=/path/to/shaep."
    local zsha; zsha=$(sha "$tmp/shaep.zip")
    [ "$zsha" = "$SHAEP_ZIP_SHA" ] || die "ShaEP zip sha256 $zsha != locked $SHAEP_ZIP_SHA"
    unzip -o -q "$tmp/shaep.zip" -d "$tmp/x"
    local b; b=$(find "$tmp/x" -type f -name 'shaep*' | head -1)
    [ -n "$b" ] || die "no shaep binary inside the archive"
    install -m 0755 "$b" "$dst"
    rm -rf "$tmp"
    ok "downloaded + zip checksum verified"
  fi
  local got; got=$(sha "$dst")
  if [ "$got" != "$SHAEP_SHA" ]; then
    [ -n "${SHAEP_SOURCE:-}" ] || die "ShaEP binary sha256 $got != locked $SHAEP_SHA"
  else
    ok "binary checksum matches versions.lock"
  fi
  "$dst" --version 2>&1 | grep -i version | head -1 | sed 's/^/    /'
}

# ---------------------------------------------------------------------------- ROSHAMBO2
install_roshambo2() {
  say "ROSHAMBO2 1.0.0  (its own conda env; CUDA-linked, cannot share an interpreter with fss)"
  local py="$HOME/.conda/envs/roshambo2/bin/python"
  [ -x "$py" ] || die "roshambo2 conda env not found at ~/.conda/envs/roshambo2"
  "$py" -c "import roshambo2; print('    roshambo2', getattr(roshambo2,'__version__','1.0.0'), 'importable')" \
    || die "roshambo2 not importable in its env"
  ok "verified in place (not rebuilt -- CUDA build)"
  printf "    NOTE: installed via pip with no recoverable commit; Methods can cite only '1.0.0'.\n"
  printf "          To pin a commit, reinstall from a git ref and update versions.lock.\n"
}

# ---------------------------------------------------------------------------- conda envs
install_envs() {
  say "conda environments (from benchmarks/comparators/envs/*.yml)"
  command -v conda >/dev/null || die "conda not on PATH. Install miniforge/miniconda first:
    https://github.com/conda-forge/miniforge"
  local created=0
  for y in "$HERE"/envs/*.yml; do
    local n; n=$(basename "$y" .yml)
    if conda env list | awk '{print $1}' | grep -qx "$n"; then
      ok "$n already exists (delete it and re-run to rebuild)"
    else
      printf "    creating %s from %s ...\n" "$n" "$(basename "$y")"
      conda env create -q -f "$y" || die "failed to create env $n"
      ok "$n created"; created=$((created+1))
    fi
  done
  printf "    %d env(s) created.\n" "$created"
  printf "    NOTE: the fss env also needs the fss package itself:\n"
  printf "          git clone https://github.com/Nate-Somers/fast_shepherd_score.git && \\\\\n"
  printf "          (cd fast_shepherd_score && git checkout 3bd9cae) && conda run -n fss pip install -e ./fast_shepherd_score\n"
  printf "    NOTE: roshambo2 additionally needs a physical NVIDIA GPU + driver (not installable via conda).\n"
}

# ---------------------------------------------------------------------------- ESP-Sim
install_espsim() {
  say "ESP-Sim 0.0.1 @ ${ESPSIM_COMMIT:0:7}"
  local dest="$PREFIX/python"
  mkdir -p "$dest"
  # --no-deps on purpose: espsim declares no install_requires and runs on the calling env's
  # rdkit/numpy/scipy. Installed beside the env rather than into it, so the pinned fss env is untouched.
  conda run -n fss pip install -q --no-deps --upgrade --target "$dest" \
      "git+${ESPSIM_REPO}@${ESPSIM_COMMIT}" || die "espsim pip install failed"
  PYTHONPATH="$dest" conda run -n fss python -c "import espsim; print(espsim.__file__)" >/dev/null \
      || die "espsim installed but not importable"
  ok "$dest/espsim  (add it to PYTHONPATH: export PYTHONPATH=$dest:\$PYTHONPATH)"
}

# ---------------------------------------------------------------------------- run
targets=("$@")
# `all` = envs first, then every tool.
if [ ${#targets[@]} -eq 1 ] && [ "${targets[0]}" = "all" ]; then
  targets=(envs align_it phesa shaep roshambo2 espsim)
fi
[ ${#targets[@]} -eq 0 ] && targets=(align_it phesa shaep roshambo2 espsim)
for t in "${targets[@]}"; do
  case "$t" in
    envs)      install_envs ;;
    align_it)  install_align_it ;;
    phesa)     install_phesa ;;
    shaep)     install_shaep ;;
    roshambo2) install_roshambo2 ;;
    espsim)    install_espsim ;;
    *) die "unknown target: $t (all|envs|align_it|phesa|shaep|roshambo2|espsim)" ;;
  esac
done

say "done -- prefix: $PREFIX"
printf "    Verify from Python:  python -m paper.comparators.paths\n"
printf "    Point the harnesses at it:  export COMPARATORS_HOME=%s\n" "$PREFIX"
