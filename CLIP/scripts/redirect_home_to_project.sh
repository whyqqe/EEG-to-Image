#!/usr/bin/env bash
# =====================================================================================
# redirect_home_to_project.sh -- stop the Cursor server + its agent worker from writing
# ANYTHING to `$HOME` (`/home/sbaiae`), by giving them a project-local HOME.
#
# WHY THIS IS NEEDED. `/home` is a 200 GB NFS volume that is 100 % full (0 B free,
# verified 2026-10-06). The Cursor harness writes one small file PER SHELL COMMAND to
#
#     $HOME/.cursor/projects/<slug>/terminals/<id>.txt
#
# and when that write fails the command never spawns at all:
#
#     ENOSPC: no space left on device, open '.../terminals/318189.txt'
#
# An earlier fix exported `CURSOR_DATA_DIR=/project/peilab/why/.cursor` in the launchers.
# That redirects the SERVER's own data, but NOT this path: the terminal/transcript directory
# is derived from `$HOME` (the agent worker runs with `HOME=/home/sbaiae`, confirmed from
# `/proc/<pid>/environ`), so the write still lands on the full volume. The only lever that
# moves it is `HOME` itself.
#
# WHAT THIS DOES. Points `HOME` at `/project/peilab/why/home` for the cursor-server process,
# so every child -- including the agent worker and the shells it spawns -- resolves `~` to a
# volume with 2.7 PB free. The launchers live under `/project` (writable, in scope); nothing
# in `/home` is created, modified or deleted.
#
# THE TRADE-OFF, HANDLED EXPLICITLY. A new HOME is an EMPTY HOME, so tools that read
# credentials from `~` (`~/.ssh`, `~/.gitconfig`, `~/.git-credentials`, `~/.netrc`) would stop
# finding them. Those are therefore SYMLINKED in from the real home -- read-only paths only.
# `~/.cache`, `~/.config` and `~/.local` are deliberately NOT symlinked: pointing them back
# at `/home` would re-introduce exactly the writes this script exists to remove. (The project
# already redirects every cache via `samclip.config`, so nothing should want them.)
#
# REVERSIBILITY. Every patched launcher is backed up in place as `*.bak-home-<stamp>`, and
# `--revert` restores them and drops the new HOME. The change only takes effect on the NEXT
# Cursor server start, i.e. after a window reload / reconnect -- a running server keeps its
# inherited HOME.
#
# USAGE
#     bash scripts/redirect_home_to_project.sh --check     # read-only: report the situation
#     bash scripts/redirect_home_to_project.sh --apply     # patch launchers + create HOME
#     bash scripts/redirect_home_to_project.sh --revert    # restore launchers
#
# After `--apply`: reload the Cursor window (or reconnect the remote) so the server restarts
# with the new HOME, then confirm with `--check`.
# =====================================================================================
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
NEW_HOME="/project/peilab/why/home"
SERVER_BIN="/project/peilab/why/.cursor-server/bin/linux-x64"
STAMP="$(date +%Y%m%d-%H%M%S)"
REAL_HOME="${REAL_HOME:-/home/sbaiae}"

# read-only credential files that a new HOME must still resolve
LINK_IN=(.ssh .gitconfig .git-credentials .netrc .gitignore_global)

log() { printf '[home-redirect] %s\n' "$*"; }

launchers() { find "${SERVER_BIN}" -type f -path '*/bin/cursor-server' 2>/dev/null | sort; }

do_check() {
  log "real HOME            : ${REAL_HOME}"
  log "target HOME          : ${NEW_HOME}"
  if [[ -d "${REAL_HOME}/.cursor/projects" ]]; then
    log "harness write target : ${REAL_HOME}/.cursor/projects/<slug>/terminals/  (this is what fails)"
  fi
  log "launchers found      : $(launchers | wc -l)"
  local n_patched=0
  while read -r f; do
    if grep -q "CROMA_HOME_REDIRECT" "${f}" 2>/dev/null; then n_patched=$((n_patched + 1)); fi
  done < <(launchers)
  log "launchers patched    : ${n_patched}"
  if [[ -d "${NEW_HOME}" ]]; then
    log "target HOME exists   : yes ($(du -sh "${NEW_HOME}" 2>/dev/null | cut -f1))"
    ls -la "${NEW_HOME}" 2>/dev/null | sed 's/^/    /' || true
  else
    log "target HOME exists   : no (run --apply)"
  fi
}

do_apply() {
  log "creating ${NEW_HOME}"
  mkdir -p "${NEW_HOME}"
  for f in "${LINK_IN[@]}"; do
    if [[ -e "${REAL_HOME}/${f}" && ! -e "${NEW_HOME}/${f}" ]]; then
      ln -s "${REAL_HOME}/${f}" "${NEW_HOME}/${f}"
      log "  linked ${f} -> ${REAL_HOME}/${f} (read-only credential path)"
    fi
  done

  local n=0
  while read -r f; do
    if grep -q "CROMA_HOME_REDIRECT" "${f}" 2>/dev/null; then
      log "  already patched: ${f}"; continue
    fi
    cp -p "${f}" "${f}.bak-home-${STAMP}"
    # insert right after the shebang so HOME is set before anything else runs
    python3 - "${f}" "${NEW_HOME}" <<'PY'
import sys
path, newhome = sys.argv[1], sys.argv[2]
lines = open(path).read().split("\n")
out, done = [], False
for i, ln in enumerate(lines):
    out.append(ln)
    if not done and ln.startswith("#!"):
        out.append("# CROMA_HOME_REDIRECT: keep every Cursor-server write off the full /home "
                   "(see CLIP/scripts/redirect_home_to_project.sh). Revert with that script's "
                   "--revert.")
        out.append('export HOME="%s"' % newhome)
        done = True
if not done:
    raise SystemExit("no shebang found; refusing to guess where to insert")
open(path, "w").write("\n".join(out))
PY
    n=$((n + 1))
    log "  patched: ${f}  (backup ${f}.bak-home-${STAMP})"
  done < <(launchers)
  log "patched ${n} launcher(s)"
  log ""
  log "NEXT: reload the Cursor window / reconnect the remote so the server restarts with the"
  log "      new HOME, then re-run:  bash scripts/redirect_home_to_project.sh --check"
}

do_revert() {
  local n=0
  while read -r f; do
    local bak
    bak="$(ls -1t "${f}".bak-home-* 2>/dev/null | head -1 || true)"
    if [[ -n "${bak}" ]]; then
      cp -p "${bak}" "${f}"
      log "restored ${f} from ${bak}"
      n=$((n + 1))
    fi
  done < <(launchers)
  log "restored ${n} launcher(s); ${NEW_HOME} left in place (delete it manually if you want)"
}

case "${1:---check}" in
  --check)  do_check ;;
  --apply)  do_apply ;;
  --revert) do_revert ;;
  *)        log "usage: $0 [--check|--apply|--revert]"; exit 2 ;;
esac
