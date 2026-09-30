"""Is the joint loss already equivalent to two separate trainings?

The claim under test: the two towers share no parameter, and each contributes to
exactly one term of the loss, so the joint objective decomposes. If that holds, then
`dL/dtheta_sem` is unaffected by whether the vae term is in the loss, and "train them
separately" would change nothing about the gradients.

Run on CPU. Small synthetic batch -- this is about the gradient algebra, not accuracy.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from epd.losses import InfoNCE, latent_mse                       # noqa: E402
from epd.model import RetrievalModel                             # noqa: E402
from epd.tokenizer import OFFICIAL_CHANNEL_ORDER                 # noqa: E402

# The real 63-channel montage, not a hardcoded subset: EEGiT's tokenizer partitions
# these into 5 anatomical regions and refuses a list with a channel outside all of
# them (Fz alone raises). Using the shipped order also keeps this check on the same
# geometry the training run uses.
CH = list(OFFICIAL_CHANNEL_ORDER)

torch.manual_seed(0)
model = RetrievalModel(
    backbone="timm:vit_b16_in21k_orig", channel_names=CH, layers=[8, 10, 12],
    n_subjects=1, d_embed=64, image_dim=64, tokenizer_kind="eegit",
    patch_size=16, n_patches_w=14, style="time-region", pool="mean",
    timm_global_pool="avg", head_kind="eegit", img_head_kind="eegit",
    fusion_mode="uniform", struct_backbone="timm:dinov3_b16",
    struct_layers=[8, 10, 12], struct_patch_size=16, struct_n_patches_w=14,
    struct_tokenizer="eegit",
    struct_cfg=dict(base_ch=16, field_ch=8, vae_ch=4, out_hw=64),
)
model.train()

B, C, T = 6, len(CH), 250
eeg = torch.randn(B, C, T)
feat = torch.randn(B, 64)
subj = torch.zeros(B, dtype=torch.long)
vae_gt = torch.randn(B, 4, 64, 64)

# Both tokenizers z-score their input and refuse to run unfitted. train.py fits them
# on the fit split; here any statistics will do, because the question is about which
# parameters a gradient reaches, not about the numbers. Done on BOTH towers so the
# comparison is not confounded by one of them being in an unfitted state.
model.encoder.tokenizer.set_norm_stats(eeg)
model.struct.encoder.tokenizer.set_norm_stats(eeg)

# Confirm the premise: the parameter sets are disjoint by name prefix.
sem_names = {n for n, _ in model.named_parameters() if not n.startswith("struct.")}
st_names = {n for n, _ in model.named_parameters() if n.startswith("struct.")}
shared = sem_names & st_names
print(f"[premise] semantic params {len(sem_names)}, structural {len(st_names)}, "
      f"shared names {len(shared)}")
assert not shared, f"the towers DO share parameters: {sorted(shared)[:5]}"

criterion = InfoNCE()

# The forward is STOCHASTIC in train mode: nn.Dropout draws from the global RNG, and
# both towers carry dropout, so two successive forward passes get different masks and
# their gradients differ by far more than any real coupling. Seeding before each pass
# is what makes the comparison about the gradient algebra instead of about the RNG.
# (Setting eval() is NOT enough: forward_all is called with training=True explicitly,
# which keeps the fusion's and the struct tower's layer/subject dropout live.)
_OP_SEED = 1234


def _forward_both():
    torch.manual_seed(_OP_SEED)
    return model.forward_all(eeg, subj, training=True)


def run(include_vae: bool) -> dict[str, torch.Tensor]:
    """Forward both towers, build the loss with or without the vae term, backprop."""
    model.zero_grad(set_to_none=True)
    out = _forward_both()
    z_e = out["z"]
    z_i = model.encode_image(feat)
    loss = criterion(z_e, z_i)
    if include_vae:
        loss = loss + 1.0 * latent_mse(out["struct"]["vae"], vae_gt)
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}


g_both = run(include_vae=True)
g_sem_only = run(include_vae=False)

missing = set(g_both) - set(g_sem_only)
if missing:
    print(f"[info] {len(missing)} params have a grad only in the joint run "
          f"(these are structural): {sorted(missing)[:3]} ...")

worst_sem, worst_sem_name = 0.0, None
for n in g_sem_only:
    if n.startswith("struct."):
        continue
    d = float((g_both[n] - g_sem_only[n]).abs().max())
    if d > worst_sem:
        worst_sem, worst_sem_name = d, n
print(f"[semantic]   max |grad_joint - grad_semantic_only| = {worst_sem:.3e}  "
      f"({worst_sem_name})")

# And the reverse: the structural tower's grad must not move when the semantic term
# is removed. Re-run with only the vae term for that comparison.
def run_vae_only() -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    out = _forward_both()
    latent_mse(out["struct"]["vae"], vae_gt).backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}


g_vae_only = run_vae_only()
worst_st, worst_st_name = 0.0, None
for n in g_vae_only:
    if not n.startswith("struct."):
        continue
    if n not in g_both:
        continue
    d = float((g_both[n] - g_vae_only[n]).abs().max())
    if d > worst_st:
        worst_st, worst_st_name = d, n
print(f"[structural] max |grad_joint - grad_vae_only|      = {worst_st:.3e}  "
      f"({worst_st_name})")

# Does the semantic tower receive ANY gradient from the vae term? It should not.
sem_leak = max(float(g_both[n].abs().max()) if n in g_both else 0.0
               for n in g_sem_only if not n.startswith("struct."))
print(f"[leak]       semantic params present in the vae-only run: "
      f"{sorted(n for n in g_vae_only if not n.startswith('struct.'))[:4]}")

tol = 1e-6
ok = worst_sem < tol and worst_st < tol
print()
print("VERDICT: the joint objective IS separable -- each tower's gradient is "
      "identical to its own term's gradient, to %g." % tol if ok else
      "VERDICT: the gradients are NOT separable; something is shared.")
sys.exit(0 if ok else 1)
