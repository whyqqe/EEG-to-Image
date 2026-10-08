# Agent Onboarding — `CLIP/`

> Scope: this document governs any agent (human or AI) working inside
> `/project/peilab/why/CLIP/`.
> Read **§1 (File-modification scope)** and **§2 (Environment)** before running
> anything. §3 covers Slurm.
> Last updated: 2026-10-02.

---

## 0. TL;DR

1. You may **only write** inside `/project/peilab/why/CLIP/` (and the shared model
   cache `/project/peilab/why/cache/` for model weights only). Everything else is
   **read-only** — including `/home/*`.
2. Always run Python with `/project/peilab/why/CLIP/.venv/bin/python` and export
   `PYTHONNOUSERSITE=1`.
3. Redirect every cache away from `$HOME` **before** importing torch / HF /
   open_clip. `/home` is **100 % full** — a single careless download crashes the
   node.
4. The login node has **no GPU**. All GPU work goes through Slurm
   (`--partition=normal --account=peilab`).

---

## 1. File-modification scope (hard rules)

This is the red line. Violating it has caused irreversible losses and job
crashes on this cluster.

### 1.1 Directories you MUST NOT write to

| Path | Reason |
|---|---|
| `/home/*` (your own home included) | The NFS home is a **200 GB volume that is 100 % full (0 B free)**. Writing there gives `No space left on device`, which crashes running jobs and can corrupt other users' sessions. Never `pip install` without `--target`, never create dotfiles, never let a library default to `~/.cache`. |
| `/project/peilab/why/` **except** `CLIP/` and `cache/` | These are other projects' working trees. **Read-only.** |
| Other projects' `.venv/` directories | Use them if the human points you at one, but never modify them. |
| `/cm/shared/apps/**`, `/etc`, `/usr`, other users' homes | System / shared software. Read-only. |

### 1.2 Directories you MAY write to

| Path | Use |
|---|---|
| `/project/peilab/why/CLIP/` | **Your project root — the main battleground.** All code, configs, logs, results go here. |
| `/project/peilab/why/cache/` | Shared model cache. Only put **HF / torch / open_clip model weights** here (see §2.4). Do **not** dump datasets or experiment artifacts here. |
| `CLIP/scratch/` | `XDG_CONFIG_HOME`, `MPLCONFIGDIR` (see §2.4). Auto-created by `samclip.config`. Safe to delete at any time — but never put results here. |
| `/tmp/clip-$SLURM_JOB_ID` **on the compute node** | `TMPDIR`. **Required by §2.4 to be node-local**, not NFS. Wiped when the job ends, so never put results here. |

Recommended layout (create it as you go):

```
CLIP/
├── AGENTS.md      # this file
├── README.md
├── src/           # library code
├── scripts/       # runnable entry points
├── slurm/         # sbatch scripts
├── configs/       # yaml / json configs
├── data/          # local data / symlinks to shared read-only data
├── outputs/       # experiment artifacts, checkpoints, result json
├── scratch/       # XDG_CONFIG_HOME / MPLCONFIGDIR (transient, deletable)
└── logs/          # text logs (Slurm .out/.err also live under outputs/slurm)
```

### 1.3 Read vs. write across project boundaries

Reading another project's data or code is **allowed**. **Modifying it is not.** If
you need a variant, **copy it into `CLIP/` first** and edit the copy. Never
`sed`/`>>`/`mv` a file outside your scope, and never `git checkout`/`reset`
inside a directory you do not own.

### 1.4 General hygiene

- **Do not edit library code while jobs that import it are running.** `src/` lives on
  NFS, shared with every queued job; a Slurm job reads the file at *its* start, not at
  submit time. Editing `src/samclip/models/backbone.py` in two steps (body first, then
  the signature) while four stage-1 jobs were already running made all four import the
  half-edited module and die with `NameError: name 'front_end' is not defined` — a
  `NameError` reported *inside a comment line*, because the jobs resolved the file
  mid-edit. The jobs' dependent evals then sat `PENDING (DependencyNeverSatisfied)`
  forever, which looked like a queue problem and was not. Finish the edit, run
  `smoke_test.py`, and only then submit — or wait for the queue to drain.
- Prefer adding files over deleting them. If you believe an out-of-scope file
 must change, **stop and ask the human**.
- Never run a recursive destructive command (`rm -rf`, `find -delete`,
  `git clean`) whose target is not strictly inside `CLIP/`. Confirm with `pwd`
  and an explicit path first.
- `/project` has **2.7 PB free (5.0 PB total, 47 % used — measured 2026-10-04)**. An
  earlier version of this line said "100 % full with only ~59 GB free"; that was stale
  and wrong, and it caused a design decision to be made on a non-constraint (a v5 plan
  proposed streaming EEG repetitions to avoid caching ~38 GB, when caching them is
  trivially affordable). Do **not** trade data fidelity for `/project` space. Still be
  tidy — an unbounded pile of per-epoch checkpoints is worth deleting because it makes a
  partial run indistinguishable from a finished one, not because of capacity.

---

## 2. Environment setup

### 2.1 The canonical interpreter

Use a **project-owned venv**:

```bash
/project/peilab/why/CLIP/.venv/bin/python
```

Create it once **from a shared base interpreter** (plain venv — do **not** use
`--system-site-packages` against another project's env, that couples the two).
The base python below lives on `/cm/shared`, so the venv resolves on compute
nodes too — see §2.2 for why that matters:

```bash
BASE_PY=/cm/shared/apps/Anaconda3/2023.09-0/bin/python3   # Python 3.11, shared
"${BASE_PY}" -m venv /project/peilab/why/CLIP/.venv
```

Then install what the project needs, keeping the cache off `$HOME` (§2.4):

```bash
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
/project/peilab/why/CLIP/.venv/bin/pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121
/project/peilab/why/CLIP/.venv/bin/pip install \
    open_clip_torch transformers timm numpy scipy scikit-learn einops tqdm pyyaml
```

If you need a one-off package without touching the venv, install it elsewhere and
prepend it to `PYTHONPATH`:

```bash
/project/peilab/why/CLIP/.venv/bin/pip install \
    --target /project/peilab/why/CLIP/.vendored <pkg>
export PYTHONPATH="/project/peilab/why/CLIP/.vendored:${PYTHONPATH}"
```

### 2.2 Using the venv under Slurm

A venv that works on the login node is not automatically usable in a Slurm job.
Observe the following rules.

**Keep the venv on shared storage.** `/project` is exported to every compute
node; `/home` is (nearly) full and `/tmp` is **node-local and wiped when the job
ends**. Therefore:

- Create the venv at `/project/peilab/why/CLIP/.venv` — **never** under
  `/home/...` or `/tmp`.
- It is fine to run `python -m venv` on the login node (it is just file writes),
  because the resulting `.venv` sits on `/project` and is visible from the GPU
  nodes.

**Create it from a base interpreter that also exists on the compute nodes.** A
venv records the absolute path of its base python in `.venv/pyvenv.cfg`, and
`bin/python` is a link into it. So the base interpreter's directory must be on
shared storage. Use the shared Anaconda python (as in §2.1):

```bash
BASE_PY=/cm/shared/apps/Anaconda3/2023.09-0/bin/python3   # NOT a login-only interpreter
"${BASE_PY}" -m venv /project/peilab/why/CLIP/.venv
```

> The venv must be created by an **existing** base interpreter — you cannot use
> `.venv/bin/python` before the venv exists.

**Do not rely on `activate` inside `sbatch`.** Slurm does **not** source
`~/.bashrc`/`~/.profile`, and `conda activate` / `source activate` may be
undefined in the job shell. Two robust options:

```bash
# (a) RECOMMENDED: skip activation, call the interpreter by absolute path
"${ROOT}/.venv/bin/python" -m clip.train --config configs/default.yaml

# (b) only if you need the venv's console scripts on PATH
source "${ROOT}/.venv/bin/activate"
python -m clip.train --config configs/default.yaml
```

**Verify the venv on a GPU node, not the login node.** Submit a 10-minute probe
so the check exercises the real environment (CUDA, NFS mount, wheel ABI):

```bash
srun --partition=normal --account=peilab --nodes=1 --ntasks=1 \
     --cpus-per-task=4 --mem=16G --gres=gpu:1 --time=00:10:00 \
     /project/peilab/why/CLIP/.venv/bin/python - <<'PY'
import sys, torch, open_clip              # noqa
print("exe   ", sys.executable)
print("torch ", torch.__version__)
print("cuda  ", torch.version.cuda, "available:", torch.cuda.is_available())
print("gpu   ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
```

Expected: `available: True` and a real GPU name. If it is `False` here, the
wheel is broken for the cluster.

**Match the torch wheel to the compute-node driver.** The login node's
`nvidia-smi` does not necessarily reflect the GPU nodes. Check the driver on a
GPU node before choosing a wheel, and pick a CUDA build whose major version is
**≤** the driver's capability:

```bash
srun --partition=normal --account=peilab --gres=gpu:1 --time=00:05:00 nvidia-smi
```

**Install-time caches also must leave `$HOME`.** `pip`/`uv` default their caches
and build dirs to `$HOME` and `/tmp`. Export the vars from §2.4 (notably
`PIP_CACHE_DIR` and `TMPDIR`) **before** creating the venv or installing
packages.

**Do not move an existing venv.** Its `bin/*` shebangs hard-code the absolute
path; relocating the directory breaks it. If you must relocate, recreate it
instead of `mv`.

**Pin what you install.** Keep a `requirements.txt` at the project root
(`pip freeze > requirements.txt`) so a job that crashed mid-week can be
reproduced exactly.

> ⚠️ Creating a venv with `--system-site-packages` that inherits another
> project's env couples the two environments: a package change in the parent
> silently changes your runs, and it still breaks if that parent is not on shared
> storage. Prefer the standalone venv above and install what you need explicitly.

### 2.3 Conda (if you prefer conda over venv)

Conda lives on shared storage. Always source it explicitly — it is **not** on
`PATH` by default and `.bashrc` is not guaranteed to run under Slurm:

```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate <your-env>
```

For a dedicated env:

```bash
conda create -n clip python=3.11 -y
```

> ⚠️ By default conda puts new envs under `/home`, which is **full** — creation
> will fail. Either keep your env **outside `/home`** by setting
> `CONDA_ENVS_PATH` / `CONDA_PKGS_DIRS` to a directory under `/project`, or prefer
> the venv in §2.1.

### 2.4 Cache redirection — mandatory

`$HOME` is full, so every tool that defaults to `~/.cache` must be redirected
**before** it is imported. Put this block at the top of **every** script and
Slurm job:

```bash
export PYTHONNOUSERSITE=1                 # ignore ~/.local packages
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=$HF_HOME
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
```

**`TMPDIR` is the one exception: it must be NODE-LOCAL, not on NFS.** This was learned
the expensive way, so the reasoning is recorded rather than the rule alone:

```bash
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"   # node-local, job-scoped
mkdir -p "${TMPDIR}"
```

`XDG_CONFIG_HOME` and `MPLCONFIGDIR` are the two that bite *later* rather than now:
nothing in this project imports `matplotlib` or a logging dashboard today, but the
moment one is added it writes a font/state cache to `~/.config` — the same full volume —
and fails at import. Closing them now costs two lines and removes a whole class of
future failure.

Why `TMPDIR` differs: pointing it at `/project` (NFS) makes every job end with ~3600
lines of

```
OSError: [Errno 16] Device or resource busy: '.nfs000000089f450aef00003343'
  ... multiprocessing/util.py::_remove_temp_dir -> shutil.rmtree
```

`multiprocessing` creates a temp dir per DataLoader worker and deletes it from an *exit
finalizer*, while files inside may still be open. On a local filesystem `unlink` on an
open file simply works; **on NFS the file is silly-renamed to `.nfs*` instead**, which
stays visible, so `rmtree` hits EBUSY and the finalizer raises. It fires during
interpreter shutdown, so results were never wrong (`rc=0`) — but it left 960 stale
`pymp-*` directories behind and buried the real log. **A log where every job emits 3600
lines of false tracebacks is a log where a real error cannot be seen**, which is the
actual cost.

An earlier version of this section claimed "our workload barely uses `TMPDIR` (the
DataLoader uses the `file_descriptor` strategy, so no temp files are written)". That was
wrong and the 3600-line tracebacks are the evidence: the sharing strategy governs how
*tensors* cross the worker boundary, and has nothing to do with `multiprocessing`'s own
per-worker temp directory, which is created regardless.

`samclip/config.py` sets all of the above itself (via `setdefault`, so an explicit env
var still wins) and pre-creates the directories — which is what makes a **login-node or
interactive** run safe, since those never see an sbatch header.

Before downloading a model, **check the shared cache first** (§4) — several
backbones are already there.

### 2.5 Quick self-test

```bash
export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
/project/peilab/why/CLIP/.venv/bin/python - <<'PY'
import torch, open_clip, transformers   # noqa
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

On the **login node** `cuda` will be `False` — that is expected (§3). Assert
CUDA availability only *inside* a Slurm job.

---

## 3. Slurm usage

### 3.1 The login node has no GPU

`torch.cuda.is_available()` is `False` on the login node. **All GPU work must be
submitted** with `sbatch` (or `srun` for short interactive jobs). Do not run
training, or heavy batch preprocessing, on the login node — it is shared.

### 3.2 Partitions and account

| Partition | Nodes | Time limit | Notes |
|---|---|---|---|
| `normal` | ~26 `dgx-*` | unlimited | **Default choice.** Has GPUs. |
| `preempt` | ~22 `dgx-*` | unlimited | **Can be preempted** — do not put long jobs here. |
| `cpu` | 2 `intel-*` | 12 h | CPU-only; for data crunching, not training. |

- Account is **`--account=peilab`** (required).
- **Exclude the unstable nodes** `dgx-09,dgx-11,dgx-17,dgx-30`.

### 3.3 Job template

```bash
#!/bin/bash
#SBATCH --job-name=clip-<task>
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/clip-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/clip-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

# --- caches (see §2.4): MUST come before any torch/HF import ---
export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
# --- TMPDIR must be NODE-LOCAL, not NFS (see §2.4) ---
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"

# --- log the environment so a failed run is diagnosable ---
echo "[slurm] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
[[ -n "${co_tenants}" ]] && echo "[slurm] shared-node co-tenants:" && echo "${co_tenants}" || true

# --- hard CUDA gate: fail fast instead of silently running on CPU for hours ---
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

"${PY}" -m clip.train --config configs/default.yaml

echo "[slurm] finished rc=$? at $(date -Is)"
```

### 3.4 Everyday commands

```bash
sbatch slurm/train.sbatch                 # submit
squeue -u "$USER"                         # your queue
squeue -p normal -w dgx-05 -h -o '%i %u %j'   # who else is on a node
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem   # after it ends
scancel <jobid>                            # cancel
sinfo                                      # partition/node status
srun --partition=normal --account=peilab --gres=gpu:1 --pty bash   # interactive GPU shell (debug)
```

### 3.5 Slurm discipline

- **Make jobs resumable.** Each stage must `[SKIP]` when its output already
  exists, so a wall-clock kill or preemption costs only the current stage.
- **Log to files, not just stdout.** After preemption, the `.out`/`.err` are your
  only evidence.
- **Print the full config at start** (checkpoint path, data hash, dims, seed,
  backbone name) so a finished run can be identified later.
- Set `--mem` to what you actually need; a wrong `--mem` kills the job *after*
  the expensive data load. Measure first with a small run.
- Prefer `--partition=normal`; use `preempt` only for short or restartable work.

---

## 4. Shared model cache (read-only)

Do not re-download what is already present. The shared cache root is
`/project/peilab/why/cache/`; point the env vars in §2.4 at it and **check it
before downloading**:

```
/project/peilab/why/cache/huggingface/hub/   # HF hub — already holds timm CLIP/SigLIP/DINOv2 backbones
/project/peilab/why/cache/open_clip/         # open_clip weights
/project/peilab/why/cache/torch/             # torch hub
```

If the exact checkpoint you want is missing, **download it into this cache**
(never into `$HOME`), then symlink or reference it from your config. `/project` is
**not** the binding constraint: 2.7 PB free (measured 2026-10-04), not the ~59 GB an
earlier revision of this file claimed. The constraint is `$HOME` (200 GB, 100 % full,
0 B free), which is why every cache variable above points away from it.

---

## 5. Pre-flight checklist

Before submitting any non-trivial job, confirm:

- [ ] You are writing **only** under `/project/peilab/why/CLIP/` (and the model
      cache). `pwd` and target paths checked.
- [ ] `PYTHONNOUSERSITE=1` and all cache vars from §2.4 are exported **before**
      Python starts.
- [ ] Interpreter is an explicit absolute path (`.../CLIP/.venv/bin/python`), not
      an `activate`-dependent bare `python` — Slurm does not source `~/.bashrc`.
- [ ] The venv has been smoke-tested **on a GPU node** (via `srun`, §2.2), not
      only on the login node.
- [ ] Slurm script has `--partition=normal --account=peilab`, the node excludes,
      a CUDA hard-gate, and logs under `CLIP/outputs/slurm/`.
- [ ] Output paths are inside `CLIP/`; nothing lands in `$HOME`. Transient state
      (`XDG_CONFIG_HOME`, `MPLCONFIGDIR`) points at `CLIP/scratch/` and `TMPDIR` at
      node-local `/tmp/clip-$SLURM_JOB_ID` (§2.4) — never `$HOME`, never NFS for `TMPDIR`.
- [ ] The job is resumable and prints its config at start.
- [ ] You have not re-downloaded models that already exist in the shared cache (§4).
