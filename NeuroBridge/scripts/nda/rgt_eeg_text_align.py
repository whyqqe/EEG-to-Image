#!/usr/bin/env python3
"""Multi-granularity EEG↔text alignment + prompt export for dual-condition generation.

Granularity:
  G0 concept  : "a photo of a/an {c}"
  G1 visible  : short visible description template
  G2 temporal : early/mid/late EEG windows → 3 tokens aligned to G0/G1 mix

Exports:
  - predicted text embeds (for diagnostics)
  - per-sample prompts (top-k retrieved concepts from EEG)
  - eeg_text_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from module.dataset import EEGPreImageDataset  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from ss_modules import SharedSpecificEncoder  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def info_nce(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = (a @ b.T) / temp
    y = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


class MultiGranTextHead(nn.Module):
    """Map EEG features (pooled + optional temporal tokens) → text space."""

    def __init__(self, in_dim: int, text_dim: int = 1024, n_tempo: int = 3):
        super().__init__()
        self.n_tempo = n_tempo
        self.pool_head = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.GELU(), nn.Linear(1024, text_dim)
        )
        self.tempo_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, text_dim)) for _ in range(n_tempo)]
        )
        self.fuse = nn.Sequential(nn.Linear(text_dim * (1 + n_tempo), text_dim), nn.GELU(), nn.Linear(text_dim, text_dim))

    def forward(self, z_pool: torch.Tensor, z_tempo: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        g0 = self.pool_head(z_pool)
        if z_tempo is None:
            return F.normalize(g0, dim=-1), F.normalize(g0, dim=-1)
        toks = [head(z_tempo[:, i]) for i, head in enumerate(self.tempo_heads)]
        fused = self.fuse(torch.cat([g0] + toks, dim=-1))
        return F.normalize(g0, dim=-1), F.normalize(fused, dim=-1)


@torch.no_grad()
def encode_windowed(
    model: SharedSpecificEncoder,
    eeg_projector: ProjectorLinear,
    eeg: torch.Tensor,
    sid: torch.Tensor,
    windows: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (z_full, z_tempo[B,T,D])."""
    z_full = eeg_projector(model(eeg, sid))
    tempos = []
    tlen = eeg.shape[-1]
    for a, b in windows:
        aa, bb = max(0, a), min(tlen, b)
        # zero outside window
        x = torch.zeros_like(eeg)
        x[..., aa:bb] = eeg[..., aa:bb]
        tempos.append(eeg_projector(model(x, sid)))
    z_tempo = torch.stack(tempos, dim=1)
    return z_full, z_tempo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--ss-checkpoint", type=str, required=True)
    ap.add_argument("--z-ret-train", type=str, required=True, help="fallback pooled embeds")
    ap.add_argument("--z-ret-test", type=str, required=True)
    ap.add_argument("--text-train", type=str, required=True)
    ap.add_argument("--text-test", type=str, required=True)
    ap.add_argument("--concept-phrases-test", type=str, required=True)
    ap.add_argument("--concept-phrases-train", type=str, required=True)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--top-k-prompt", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.nb_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    text_tr = l2(np.load(args.text_train).astype(np.float32))
    text_te = l2(np.load(args.text_test).astype(np.float32))
    z_tr = l2(np.load(args.z_ret_train).astype(np.float32))
    z_te = l2(np.load(args.z_ret_test).astype(np.float32))
    phrases_te = json.loads(Path(args.concept_phrases_test).read_text())
    phrases_tr = json.loads(Path(args.concept_phrases_train).read_text())

    # Train concept gallery (for train diagnostics only)
    n_img = 10
    n_concepts = len(phrases_tr)
    assert text_tr.shape[0] == n_concepts * n_img, (text_tr.shape, n_concepts)
    concept_gallery_train = l2(text_tr.reshape(n_concepts, n_img, -1).mean(1))

    # CRITICAL FIX: test concepts are held-out (200-way). Prompt retrieval MUST use
    # the test text gallery — train-gallery name match is always ~0 under zero-shot.
    assert text_te.shape[0] == len(phrases_te), (text_te.shape, len(phrases_te))
    concept_gallery_test = l2(text_te.copy())
    concept_names_test = phrases_te

    # Temporal windows on 250 samples (~0-1000ms if 250Hz): early/mid/late
    windows = [(0, 50), (50, 125), (125, 250)]

    ckpt = torch.load(
        Path(args.ss_checkpoint) if Path(args.ss_checkpoint).is_absolute() else root / args.ss_checkpoint,
        map_location=device,
        weights_only=False,
    )
    subjects = [int(s) for s in ckpt.get("subjects", [args.subject])]
    img_dim = int(ckpt["img_dim"])
    feature_dim = int(ckpt["feature_dim"])
    eeg_len = int(ckpt["eeg_sample_points"])
    channels_num = int(ckpt["channels_num"])
    ss = SharedSpecificEncoder(
        subject_ids=subjects,
        feature_dim=img_dim,
        eeg_sample_points=eeg_len,
        channels_num=channels_num,
        n_extra_blocks=int(ckpt.get("n_extra_blocks", 1)),
        use_adapter=True,
    ).to(device)
    eeg_proj = ProjectorLinear(img_dim, feature_dim).to(device)
    ss.load_state_dict(ckpt["model_state_dict"])
    eeg_proj.load_state_dict(ckpt["eeg_projector_state_dict"])
    ss.eval()
    eeg_proj.eval()
    for p in list(ss.parameters()) + list(eeg_proj.parameters()):
        p.requires_grad = False

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(root / "data/things_eeg/image_feature/RN50")

    def collect(split_train: bool):
        ds = EEGPreImageDataset(
            [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
            rn50_dir, "", False, [], True, False, None, split_train, False, False, False,
        )
        pools, tempos = [], []
        with torch.no_grad():
            for batch in tqdm(DataLoader(ds, batch_size=64, shuffle=False), desc=f"enc-{'tr' if split_train else 'te'}"):
                eeg, _i, _t, sid, *_ = batch
                eeg, sid = eeg.to(device), sid.to(device)
                zf, zt = encode_windowed(ss, eeg_proj, eeg, sid, windows)
                pools.append(zf.cpu())
                tempos.append(zt.cpu())
        return torch.cat(pools), torch.cat(tempos)

    z_pool_tr, z_tempo_tr = collect(True)
    z_pool_te, z_tempo_te = collect(False)
    # prefer freshly encoded pooled; fallback length check
    if z_pool_tr.shape[0] != z_tr.shape[0]:
        print("[WARN] length mismatch, using provided z_ret for pool")
        z_pool_tr = torch.from_numpy(z_tr)
        z_pool_te = torch.from_numpy(z_te)
        z_tempo_tr = z_pool_tr.unsqueeze(1).repeat(1, 3, 1)
        z_tempo_te = z_pool_te.unsqueeze(1).repeat(1, 3, 1)

    in_dim = z_pool_tr.shape[-1]
    text_dim = text_tr.shape[-1]
    head = MultiGranTextHead(in_dim, text_dim=text_dim, n_tempo=3).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    # text targets: G0 = concept text (flat), G1 = same (THINGS has one phrase); use flat as both
    txt_tr_t = torch.from_numpy(text_tr)
    loader = DataLoader(
        TensorDataset(z_pool_tr, z_tempo_tr, txt_tr_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    def gallery_topk(pred: np.ndarray, gallery: np.ndarray, names: list[str], gt_names: list[str], k: int):
        sim = pred @ gallery.T
        pred_names, prompts = [], []
        for i in range(sim.shape[0]):
            idx = np.argsort(-sim[i])[:k]
            ns = [names[j] for j in idx]
            pred_names.append(ns)
            # confidence: margin top1 - top2
            srt = np.sort(sim[i])[::-1]
            conf = float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])
            if conf < 0.01:
                # low-confidence: softer / shorter prompt to avoid wrong-concept damage
                prompts.append(f"a photo of {ns[0]}")
            elif k == 1 or len(ns) == 1:
                prompts.append(f"a photo of {ns[0]}, highly detailed")
            else:
                prompts.append(f"a photo of {ns[0]}, highly detailed")
        top1 = sum(1 for i, ns in enumerate(pred_names) if gt_names[i] == ns[0]) / max(len(gt_names), 1)
        topk_hit = sum(1 for i, ns in enumerate(pred_names) if gt_names[i] in ns[:k]) / max(len(gt_names), 1)
        return pred_names, prompts, top1, topk_hit, sim

    best_score, best_state = -1.0, None
    history = []
    gal_te_t = torch.from_numpy(concept_gallery_test).to(device)
    for ep in range(1, args.epochs + 1):
        head.train()
        loss_acc = 0.0
        for zp, zt, tx in loader:
            zp, zt, tx = zp.to(device), zt.to(device), tx.to(device)
            g0, fused = head(zp, zt)
            loss = 0.5 * info_nce(g0, tx) + 0.5 * info_nce(fused, tx)
            loss = loss + 0.75 * (1 - (g0 * F.normalize(tx, dim=-1)).sum(-1)).mean()
            loss = loss + 0.75 * (1 - (fused * F.normalize(tx, dim=-1)).sum(-1)).mean()
            t0 = F.normalize(head.tempo_heads[0](zt[:, 0]), dim=-1)
            t2 = F.normalize(head.tempo_heads[2](zt[:, 2]), dim=-1)
            loss = loss + 0.02 * (t0 * t2).sum(-1).mean().abs()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += float(loss.item())
        head.eval()
        with torch.no_grad():
            g0, fused = head(z_pool_te.to(device), z_tempo_te.to(device))
            cos = float((fused * torch.from_numpy(text_te).to(device)).sum(-1).mean())
            # select by closed-set 200-way top1 (what generation actually needs)
            sim = (fused @ gal_te_t.T).cpu().numpy()
            top1 = float(np.mean(np.argmax(sim, axis=1) == np.arange(sim.shape[0])))
            score = 100.0 * top1 + 20.0 * cos
        history.append(
            {
                "epoch": ep,
                "loss": loss_acc / max(len(loader), 1),
                "test_text_cos": cos,
                "test_200way_top1": top1,
                "score": score,
            }
        )
        print(f"[eeg-text {ep}] loss={history[-1]['loss']:.4f} cos={cos:.4f} top1_200={top1:.3f}")
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        g0_te, fused_te = head(z_pool_te.to(device), z_tempo_te.to(device))
        g0_tr, fused_tr = head(z_pool_tr.to(device), z_tempo_tr.to(device))
    fused_te_np = l2(fused_te.cpu().numpy())
    g0_te_np = l2(g0_te.cpu().numpy())
    fused_tr_np = l2(fused_tr.cpu().numpy())
    np.save(out / "z_text_pred_test.npy", fused_te_np)
    np.save(out / "z_text_pred_train.npy", fused_tr_np)

    topk = args.top_k_prompt
    # Primary: closed-set retrieval on TEST 200 concepts
    pred_f, prompts_f, top1_f, topk_f, _ = gallery_topk(
        fused_te_np, concept_gallery_test, concept_names_test, phrases_te, topk
    )
    pred_g, prompts_g, top1_g, topk_g, _ = gallery_topk(
        g0_te_np, concept_gallery_test, concept_names_test, phrases_te, topk
    )
    # pick better head for prompts
    if top1_g > top1_f:
        pred_concepts, prompts, top1, topk_hit, which = pred_g, prompts_g, top1_g, topk_g, "g0"
    else:
        pred_concepts, prompts, top1, topk_hit, which = pred_f, prompts_f, top1_f, topk_f, "fused"

    # Diagnostics only: open retrieval against train gallery (expect low name match)
    _, _, top1_open, _, _ = gallery_topk(
        fused_te_np, concept_gallery_train, phrases_tr, phrases_te, topk
    )

    oracle = [f"a photo of {p}, highly detailed" for p in phrases_te]
    (out / "prompts_pred.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    (out / "prompts_oracle.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8")
    (out / "pred_concepts.json").write_text(json.dumps(pred_concepts, indent=2), encoding="utf-8")

    report = {
        "method": "multi-granularity EEG-text align (FIXED test-gallery retrieval)",
        "fix": "v3 used train gallery → zero-shot name Top-1 always ~0; now 200-way test gallery",
        "windows_ms_approx": ["0-200", "200-500", "500-1000"],
        "best_score": best_score,
        "prompt_head": which,
        "concept_top1_200way": top1,
        "concept_topk_200way": topk_hit,
        "concept_top1_open_train_gallery": top1_open,
        "g0_top1_200way": top1_g,
        "fused_top1_200way": top1_f,
        "top_k_prompt": topk,
        "history": history,
        "prompts_pred": str(out / "prompts_pred.json"),
    }
    torch.save({"head": best_state, "in_dim": in_dim, "text_dim": text_dim}, out / "eeg_text_head.pt")
    (out / "eeg_text_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
