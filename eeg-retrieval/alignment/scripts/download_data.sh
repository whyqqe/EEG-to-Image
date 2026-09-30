#!/bin/bash
# ============================================================================
# Download MS-COCO (Karpathy split) + Flickr30k for post-hoc alignment study
# ----------------------------------------------------------------------------
# Discipline (per HANDOFF.md §2):
#   - write ONLY under /project/peilab/why/eeg-retrieval/
#   - redirect all caches away from /home
#   - every stage is resumable: skip if product already exists
#   - log to file, not just stdout
# ============================================================================
set -uo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
DATA=$ROOT/data
LOG=$ROOT/logs
mkdir -p "$DATA/coco" "$DATA/flickr30k" "$LOG"

# --- cache redirection (MANDATORY: /home quota is tiny) --------------------
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR=/project/peilab/why/eeg-retrieval/alignment/data/tmp
mkdir -p "$TMPDIR"

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

# ============================== 1. Karpathy splits ==========================
# caption_datasets.zip contains dataset_coco.json + dataset_flickr30k.json
# (the canonical Karpathy train/val/test splits with 5 captions per image)
KF=$DATA/caption_datasets.zip
if [ -f "$DATA/dataset_coco.json" ] && [ -f "$DATA/dataset_flickr30k.json" ]; then
    say "[SKIP] karpathy split jsons already present"
else
    say "downloading Karpathy split annotations ..."
    curl -sSL -C - --retry 5 --retry-delay 5 -o "$KF" \
        https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip \
        || { say "[FATAL] karpathy download failed"; exit 1; }
    unzip -o -q "$KF" -d "$DATA" && say "[OK] extracted karpathy jsons"
fi

# ============================== 2. COCO images ==============================
# train2014 ~13.5 GB, val2014 ~6.6 GB. aria2c with 16 connections + resume.
dl_zip() {  # $1 = url basename e.g. train2014, $2 = expected dir
    local name="$1" zipp="$DATA/coco/$1.zip"
    if [ -d "$DATA/coco/$1" ] && [ "$(ls -A "$DATA/coco/$1" 2>/dev/null | head -1)" ]; then
        say "[SKIP] $1 already extracted"; return 0
    fi
    if [ ! -f "$zipp" ]; then
        say "downloading $1.zip ..."
        aria2c -c -x16 -s16 -k1M --file-allocation=none --summary-interval=60 \
               --console-log-level=warn --dir="$DATA/coco" -o "$1.zip" \
               "http://images.cocodataset.org/zips/$1.zip" \
            || { say "[FATAL] $1 download failed"; return 1; }
    fi
    say "extracting $1.zip ..."
    unzip -o -q "$zipp" -d "$DATA/coco" && say "[OK] $1 extracted"
    # free the zip to save space (5.8T avail, but be tidy)
    rm -f "$zipp"
}

# ============================== 3. Flickr30k ===============================
# Primary mirror: HuggingFace nlphuji/flickr30k (images/ contains the jpgs)
dl_flickr() {
    # HF repo nlphuji/flickr30k contains:
    #   flickr30k-images.zip       (~4.5 GB, 31,783 jpgs)
    #   flickr_annotations_30k.csv (captions)
    local base="https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main"
    if [ -d "$DATA/flickr30k/flickr30k-images" ] || [ -d "$DATA/flickr30k/images" ]; then
        say "[SKIP] flickr30k images present"
    else
        say "downloading flickr30k images from HuggingFace mirror ..."
        aria2c -c -x8 -s8 -k1M --file-allocation=none --summary-interval=60 \
               --console-log-level=warn --dir="$DATA/flickr30k" -o flickr30k-images.zip \
               "$base/flickr30k-images.zip" \
            || { say "[FATAL] flickr30k download failed"; return 1; }
        say "extracting flickr30k-images.zip ..."
        unzip -o -q "$DATA/flickr30k/flickr30k-images.zip" -d "$DATA/flickr30k" \
            && say "[OK] flickr30k extracted"
        rm -f "$DATA/flickr30k/flickr30k-images.zip"
    fi
    # captions (small)
    if [ ! -f "$DATA/flickr30k/flickr_annotations_30k.csv" ]; then
        say "downloading flickr30k annotations ..."
        curl -sSL -C - --retry 5 -o "$DATA/flickr30k/flickr_annotations_30k.csv" \
            "$base/flickr_annotations_30k.csv" && say "[OK] flickr30k annotations"
    fi
}

# ------------------------------- orchestration -----------------------------
say "=== START ==="
say "disk: $(df -h "$DATA" | tail -1)"

# small first, so the pipeline can be validated while COCO trickles in
dl_flickr
dl_zip val2014                    # 6.6 GB, needed for test set
dl_zip train2014                  # 13.5 GB, needed for the 80k-pair condition

# HF fallback for flickr30k if the zip mirror failed
if [ ! -d "$DATA/flickr30k/flickr30k-images" ] && [ ! -d "$DATA/flickr30k/images" ]; then
    if [ -f "$DATA/flickr30k/flickr30k-images.zip" ]; then
        say "extracting flickr30k zip ..."
        unzip -o -q "$DATA/flickr30k/flickr30k-images.zip" -d "$DATA/flickr30k" \
            && say "[OK] flickr30k extracted"
    fi
fi

say "=== DONE ==="
say "coco/train2014: $(ls "$DATA/coco/train2014" 2>/dev/null | wc -l) files"
say "coco/val2014:   $(ls "$DATA/coco/val2014" 2>/dev/null | wc -l) files"
say "flickr30k:      $(find "$DATA/flickr30k" -name '*.jpg' 2>/dev/null | wc -l) files"
df -h "$DATA" | tail -1
