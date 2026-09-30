#!/usr/bin/env python3
"""GEM stage 5 -- the ANALYSIS the generation rows cannot answer by themselves.

Runs after evaluation, offline, on arrays that are already on disk.  It exists
because "the images scored well" is not evidence for a three-tower architecture.
Three questions have to be answered separately, and none of them is a FID number:

  Q1 ARE THE THREE TOWERS ACTUALLY THREE?
     The 3x3 cross-prediction matrix (fitted and scored inside `gem_train.py`,
     where the target embeddings already exist) is re-read here and turned into a
     verdict.  The test: for each tower's OWN target, does any OTHER tower's
     activation predict it about as well?  If `sem` predicts the CLIP-image target
     as well as `clip` does, the CLIP tower is a relabelling and the honest report
     says so -- which also means the generation row that leans on it is not
     evidence for anything.

  Q2 WHICH DESCRIPTION FIELDS ARE GROUNDED?
     A field's `same_concept_nn_rate` (is a test row's field embedding nearest to
     another test row of the SAME concept?) against its chance rate.  This is the
     right instrument because the train and test CONCEPT SETS ARE DISJOINT: a
     classifier cannot be scored across them, so a retrieval rate is the only
     honest measurement available.  A field at chance is decoration.

  Q3 DOES THE DECODED TEXT ACTUALLY CONTAIN THE CONCEPT?
     The semantic tower emits a STRING, so the concept word is directly checkable:
     the fraction of decoded prompts containing the row's concept, against the base
     rate at which the same word appears anywhere in the 200-row decode.  This is
     the interpretable version of "the text tower works", and it needs no oracle:
     the concept label is the standard retrieval supervision.
     The `unrel` row is also checked here, because the CLIP-text score it was
     generated for cannot separate "the condition was read" from "the prompt was
     the whole story".

  Q4 THE ATTRIBUTION TABLE, WITH ITS CONTROLS SIDE BY SIDE.
     `gem_ll_self` is meaningless on its own.  It is printed next to `gem_unrel`
     (text right, condition from another row), `gem_stat` (constant condition),
     `gem_nofront` (front end off) and `gem_oracle` (ceiling).  The verdict is
     stated as a comparison, and a row that does not beat its control is reported
     as not beating it rather than being dropped.

WHAT THIS FILE DELIBERATELY DOES NOT DO
---------------------------------------
It does not re-score images, and it does not touch the test images' own
descriptions.  Every number comes from `gem_report.json`, `acts_*.npz`, the
decoded prompts the model itself produced, and the seven-metric eval JSONs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

METS = ("pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav")


def l2(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)


def same_concept_nn(pred: np.ndarray, labels: list[str], csls_k: int = 10) -> dict:
    """Top-1 retrieval rate of a same-concept row, and the rate chance implies.

    Reported TWICE: raw cosine, and CSLS-corrected.  The correction is not a tweak
    -- it is a different estimator of the same quantity, and the EEG-to-image
    literature reports a large gap between them (78.1% -> 86.4% Top-1 in the
    neural-visibility work cited in the header).  Reporting only the raw number
    would understate the method AND hide what part of the score is hubness.

    CSLS (Conneau et al.) is `2*cos(i,j) - r(i) - r(j)`, where `r(i)` is the mean of
    i's top-k similarities to the rest.  A HUB row -- one that sits near everything,
    which is exactly what a partially collapsed condition looks like -- has a high
    `r`, so subtracting it removes the advantage that comes from being central
    rather than from being right.  `mean_r` is therefore also a direct read on how
    much of the row-to-row structure is hub structure.
    """
    Z = l2(np.asarray(pred, dtype=np.float32))
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    lab = np.asarray(labels)
    rate = float((lab[S.argmax(1)] == lab).mean())
    cnt = {c: int((lab == c).sum()) for c in set(labels)}
    chance = float(sum(v * (v - 1) for v in cnt.values())
                   / max(len(labels) * (len(labels) - 1), 1))
    # ---- CSLS
    k = int(min(csls_k, S.shape[1] - 1))
    r = np.sort(S, axis=1)[:, -k:].mean(1) if k > 0 else np.zeros(len(S))
    Sc = 2.0 * S - r[:, None] - r[None, :]
    rate_csls = float((lab[np.argmax(Sc, 1)] == lab).mean())
    return {"same_concept_nn_rate": rate, "chance": chance, "lift": rate - chance,
            "same_concept_nn_rate_csls": rate_csls, "lift_csls": rate_csls - chance,
            "csls_k": k, "mean_r": float(r.mean()), "sd_r": float(r.std()),
            "note": ("rate is raw cosine top-1; rate_csls subtracts the k-NN local "
                     "scale, which removes the advantage a HUB row gets from being "
                     "close to everything instead of close to its own concept")}


def concept_hit_rate(prompts: list[str], concepts: list[str]) -> dict:
    """Fraction of decoded prompts that contain the row's concept word.

    Case-insensitive whole-word matching, with the concept's own underscore-to-space
    form.  The control is the BASE RATE: how often that same word appears anywhere
    in the whole 200-row decode.  A word that every prompt mentions anyway (e.g. a
    concept whose name is a common noun the decoder emits by prior, like "dog")
    would otherwise look like a successful decode.
    """
    low = [p.lower() for p in prompts]
    hits, base = 0, 0
    for p, c in zip(low, concepts):
        w = c.replace("_", " ").lower().strip()
        if not w:
            continue
        pat = re.compile(r"\b" + re.escape(w) + r"\b")
        hits += int(bool(pat.search(p)))
        occ = sum(1 for q in low if pat.search(q))
        base += occ / max(len(low), 1)
    n = max(len(concepts), 1)
    return {"hit_rate": hits / n, "base_rate": base / n,
            "lift": hits / n - base / n,
            "note": ("hit_rate is per-row containment of the row's own concept word; "
                     "base_rate is the same word's average rate over all decoded rows, "
                     "so a word the decoder emits by prior is netted out")}


def load_eval(p: Path) -> dict | None:
    if not p.is_file() or p.stat().st_size == 0:
        return None
    d = json.load(open(p))
    return d.get("metrics", d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(NB_ROOT / "outputs/gem/sub-08"))
    ap.add_argument("--sid", type=int, default=8)
    ap.add_argument("--captions-test", type=str,
                    default=str(NB_ROOT / "outputs/g2/captions/captions_test.jsonl"))
    ap.add_argument("--arms", nargs="+", default=["full", "nofront", "noise"])
    ap.add_argument("--rows", nargs="+",
                    default=["gem_ll_self", "gem_ll_rawcal", "gem_ll_noprompt",
                             "gem_sem_noprompt", "gem_generic",
                             "gem_unrel", "gem_swap", "gem_stat", "gem_noise",
                             "gem_oracle"])
    ap.add_argument("--report", type=str, default="")
    args = ap.parse_args()

    OUT = Path(args.out)
    sd = f"{args.sid:02d}"
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    caps = [json.loads(l) for l in Path(args.captions_test).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    con_te = [Path(c["path"]).parent.name.split("_", 1)[1].replace("_", " ")
              for c in caps]

    # ============================================================ Q1 towers
    say("=" * 78)
    say("Q1  ARE THE THREE TOWERS ACTUALLY THREE?  (cross-prediction, TRAIN-fit/TEST-scored)")
    say("=" * 78)
    for arm in args.arms:
        p = OUT / arm / "gem_report.json"
        if not p.is_file():
            say(f"  [{arm}] no report"); continue
        d = json.load(open(p))
        xm = d.get("tower_attribution", {})
        rd = d.get("tower_redundancy", {})
        say(f"  [{arm}] test cosine, source -> target")
        say(f"    {'':<14}" + "".join(f"{t:>12}" for t in ("txt", "img", "vae")))
        for sn in ("sem", "clip", "vae"):
            say(f"    src {sn:<10}" + "".join(
                f"{xm.get(f'{sn}->{t}', {}).get('test_cos', float('nan')):>12.4f}"
                for t in ("txt", "img", "vae")))
        for k, v in rd.items():
            delta = v["best_other_minus_own"]
            verdict = ("REDUNDANT: another tower predicts its own target better"
                       if delta > 0 else "distinct: its own activation is the best predictor")
            say(f"    {k:<12} own {v['own']:.4f} | best other "
                f"{max(v['others'].values()):.4f} | delta {delta:+.4f} -> {verdict}")
        fg = d.get("field_grounding", {})
        say(f"  [{arm}] field grounding (same-concept top-1 rate vs chance):")
        for g, m in fg.items():
            if g.startswith("_"):
                continue
            say(f"    {g:<10} {m['same_concept_nn_rate']:.4f} "
                f"(CSLS {m.get('same_concept_nn_rate_csls', float('nan')):.4f}) vs "
                f"{m['chance']:.4f} (lift {m['lift']:+.4f}) grounded={m['grounded']} | "
                f"hub mean_r {m.get('hub_mean_r', float('nan')):.4f}")
        for k in ("_fused_condition", "_prompt_pool"):
            m = fg.get(k)
            if m:
                # `nn_same_concept` returns `same_concept_nn_rate`; this read `rate`,
                # which does not exist, so the whole [5] analysis stage died with
                #     KeyError: 'rate'
                # AFTER the evaluation JSONs had already been written.  The metrics
                # survived; the summary did not.
                say(f"    {k:<18} {m['same_concept_nn_rate']:.4f} "
                    f"(CSLS {m.get('same_concept_nn_rate_csls', float('nan')):.4f}) vs "
                    f"chance {m['chance']:.4f} (lift {m['lift']:+.4f})")
        f = d.get("frozen_condition_check", {})
        say(f"  [{arm}] frozen-condition check: pool->CLIP-text cos "
            f"{f.get('pool_to_clip_text_cos', float('nan')):.4f} | clip tower cos "
            f"{f.get('clip_tower_cos', float('nan')):.4f} | row-identity "
            f"{f.get('row_identity_acc_pool', float('nan')):.4f} "
            f"(chance {f.get('chance_row_identity', float('nan')):.4f})")
        say(f"  [{arm}] trained={d.get('trained')} grad_skips={d.get('grad_skips')}")
        cc = d.get("condition_concentration", {})
        if cc:
            say(f"  [{arm}] condition concentration: c_self {cc.get('c_self_predicted', float('nan')):.4f} "
                f"(1.0000 = the condition is a CONSTANT and cannot be row-specific)")
        # ---- the GVM mechanism verdicts.  Printed per arm and BEFORE any metric, so
        # a mechanism that never engaged is visible as such instead of being read as
        # a null result about the architecture.
        a2 = d.get("M2_anchor_ladder")
        if a2:
            say(f"  [{arm}] M2 granularity ladder: vocab {a2['n_vocab']} "
                f"budget {a2.get('total_per_prompt', '?')} words/prompt "
                f"(cap {a2['topk_per_field']}/field) | prompt unique "
                f"{a2['prompt_unique']}/{a2['prompt_n']} "
                f"({100 * a2['prompt_unique'] / max(a2['prompt_n'], 1):.1f}%) "
                f"jaccard {a2['prompt_jaccard']:.4f} "
                f"| anchor recall " + " ".join(
                    f"{g}={a2['per_field_recall'][g]:.3f}"
                    for g in ("overall", "subject", "background", "detail")))
            _tw = a2.get("top_informative_words")
            if _tw:
                say(f"    vocabulary (mutual information with the concept, TRAIN only): "
                    f"{' '.join(_tw[:16])}")
            if a2["prompt_jaccard"] > 0.9:
                say(f"    [WARN] the assembled prompt is nearly CONSTANT "
                    f"(jaccard {a2['prompt_jaccard']:.3f}). The text condition cannot "
                    f"be row-specific, so `gem_swap` is uninterpretable.")
            say(f"    NOTE the prompt is AUXILIARY. The generation condition is the "
                f"embedding-level alignment (`conds.sem`), which generates no words; "
                f"`gem_sem_noprompt` runs it with NO text prompt at all, and "
                f"`gem_ll_noprompt` does the same for the main condition. Read the "
                f"generation table before concluding anything about the text path.")
        v2 = d.get("M3_neural_visibility")
        if v2:
            say(f"  [{arm}] M3 neural-visibility A/B (held-in): direct-1024 "
                f"{v2['cos_direct_1024']:.4f} vs penultimate-1280->proj "
                f"{v2['cos_penultimate_1280_through_proj']:.4f} -> "
                f"{v2['chosen']}, margin {v2['margin']:.4f}")
        r2 = d.get("M4_arbitration")
        if r2:
            say(f"  [{arm}] M4 learned arbitration: strength "
                f"{r2['alpha_mean']:.4f}+-{r2['alpha_sd']:.4f} "
                f"[{r2['alpha_min']:.3f},{r2['alpha_max']:.3f}] | ip_scale "
                f"{r2['ipscale_mean']:.4f}+-{r2['ipscale_sd']:.4f} "
                f"[{r2['ipscale_min']:.3f},{r2['ipscale_max']:.3f}]")
            # CALIBRATION, not corr(alpha, r_est).  The latter is +-1.0 by
            # construction -- alpha is an affine function of r_est -- so it reports a
            # design decision as if it were a measurement.  What M4 claims is that the
            # head can tell PER TRIAL whether this row's semantic or structural
            # read-out is the more trustworthy one, and that is "does r_est track the
            # reliability ACHIEVED on the same row", with a permutation as the null.
            if "corr_r_est_sem_vs_achieved" in r2:
                say(f"    calibration r_est vs ACHIEVED reliability: "
                    f"sem {r2['corr_r_est_sem_vs_achieved']:+.4f} "
                    f"(permuted null {r2['corr_r_est_sem_vs_permuted']:+.4f}) | "
                    f"str {r2['corr_r_est_str_vs_achieved']:+.4f} "
                    f"(permuted null {r2['corr_r_est_str_vs_permuted']:+.4f}) "
                    f"-- 0.0 means the per-row parameter moves for reasons unrelated "
                    f"to the row")
            if r2["alpha_sd"] < 0.005:
                say(f"    [WARN] M4 is INERT here (sd {r2['alpha_sd']:.5f}): the head "
                    f"predicts the same reliability for every trial, so the fixed and "
                    f"arbitrated arms differ only by noise and M4 is UNTESTED rather "
                    f"than refuted.")
            elif min(r2.get("corr_r_est_sem_vs_achieved", 1.0),
                     r2.get("corr_r_est_str_vs_achieved", 1.0)) < 0.05:
                say(f"    [WARN] M4 is NOISE here: alpha still spreads "
                    f"(sd {r2['alpha_sd']:.4f}) but the predicted reliability does not "
                    f"track achieved reliability, so `gem_arb` vs `gem_fixalpha` "
                    f"measures sampling noise rather than adaptation.")
        s2 = d.get("innovation4_weight_schedule", {})
        if s2.get("applied_epoch") is not None:
            say(f"  [{arm}] innov4 info-weighted schedule at ep "
                f"{s2['applied_epoch']}: measured recoverable R2 "
                + " ".join(f"{k}={vv:.4f}" for k, vv in s2["r2"].items())
                + " -> multipliers "
                + " ".join(f"{k}={vv:.3f}" for k, vv in s2["mult"].items()))
    say()

    # ============================================================ Q3 prompts
    say("=" * 78)
    say("Q3  DOES THE DECODED TEXT CONTAIN THE CONCEPT?  (semantic tower as a STRING)")
    say("=" * 78)
    for arm in args.arms:
        pf = OUT / arm / "prompts" / "prompts_self.json"
        if not pf.is_file():
            say(f"  [{arm}] no decoded prompts"); continue
        pr = json.loads(pf.read_text(encoding="utf-8"))
        if len(pr) != len(con_te):
            say(f"  [{arm}] {len(pr)} prompts vs {len(con_te)} test rows: skipped")
            continue
        h = concept_hit_rate(pr, con_te)
        say(f"  [{arm}] concept word in decoded prompt: {h['hit_rate']:.4f} "
            f"(base rate {h['base_rate']:.4f}, lift {h['lift']:+.4f})")
        say(f"    e.g. {pr[0][:120]}")
    say()

    # ============================================================ Q4 attribution
    say("=" * 78)
    say("Q4  ATTRIBUTION: EVERY ROW NEXT TO ITS CONTROL  (official seven metrics)")
    say("=" * 78)
    table: dict[str, dict] = {}
    for tag in args.rows:
        m = load_eval(OUT / "eval" / f"s{sd}_{tag}.json")
        if m:
            table[tag] = m
    for ref, fname in (("sdedit_ll (HCMA ref)", "s00_sdedit_ll.json"),
                       ("g3f_selfgate (ref)", "s00_g3f_ll_selfgate.json")):
        m = load_eval(OUT / "eval" / fname)
        if m:
            table[ref] = m

    if not table:
        say("  no eval JSONs yet; run this again after stage [4]")
    else:
        say(f"  {'row':<22}" + "".join(f"{k:>10}" for k in METS))
        for tag, m in table.items():
            say(f"  {tag:<22}" + "".join(
                f"{m[k]:>10.4f}" if m.get(k) is not None else f"{'-':>10}"
                for k in METS))
        say()
        # The verdicts.  Each is a COMPARISON; a row that does not beat its control
        # is reported as such.  `unrel` is the sharpest control: the prompt is the
        # right one, only the condition is somebody else's, so anything it scores
        # above chance is attributable to the TEXT, not to the EEG condition.
        say("  VERDICTS (a row is only evidence if it beats its own control):")
        for metric in ("clip", "inception", "swav", "alex5"):
            a, b = table.get("gem_ll_self"), table.get("gem_unrel")
            c, d_ = table.get("gem_ll_self"), table.get("gem_stat")
            e, f_ = table.get("gem_ll_self"), table.get("gem_nofront")
            g, h = table.get("gem_oracle"), table.get("gem_ll_self")
            def dd(x, y):
                if not x or not y:
                    return float("nan")
                return (x.get(metric) or float("nan")) - (y.get(metric) or float("nan"))
            say(f"    {metric}: self-unrel {dd(a, b):+.4f} | self-stat {dd(c, d_):+.4f} "
                f"| self-nofront {dd(e, f_):+.4f} | oracle-self {dd(g, h):+.4f}")
        say("    reading: self-unrel > 0 means the EEG CONDITION carries concept "
            "information beyond the prompt; oracle-self is the headroom that remains.")
        # ---- M4's own comparison, kept separate because it is the ONE ablation where
        # both arms share a model, a condition, a prompt and an init, and differ only
        # in whether the per-row arbitration is read.  A positive gap is therefore
        # about the mechanism and not about anything upstream of generation.
        a, b = table.get("gem_arb"), table.get("gem_fixalpha")
        if a and b:
            say("  M4 ABLATION (same model / condition / prompt / init; only the "
                "per-row arbitration differs):")
            for metric in ("clip", "inception", "swav", "alex5"):
                x, y = a.get(metric), b.get(metric)
                if x is None or y is None:
                    continue
                say(f"    {metric}: arb {x:.4f} vs fixalpha {y:.4f} "
                    f"({x - y:+.4f})")
            say("    reading: a gap inside the eval's own noise means the per-row "
                "arbitration did not help, which is a result about the mechanism, not "
                "about the architecture.")
        else:
            say("  M4 ABLATION: `gem_arb` and/or `gem_fixalpha` absent, so M4 is "
                "UNTESTED.  That is not the same as M4 having no effect.")
        # ---- M3's ablation, same logic one level up
        a, b = table.get("gem_ll_self"), table.get("gem_layer_loser")
        if a and b:
            say("  M3 ABLATION (same model, only the T3 layer differs):")
            for metric in ("clip", "inception", "swav", "alex5"):
                x, y = a.get(metric), b.get(metric)
                if x is None or y is None:
                    continue
                say(f"    {metric}: chosen layer {x:.4f} vs loser {y:.4f} "
                    f"({x - y:+.4f})")

    say()
    say("=" * 78)
    say("WHAT THIS FILE CANNOT TELL YOU: whether the images are good.  Every number "
        "above is either a retrieval statistic or a control comparison; the seven "
        "metrics are printed only so a row and its control can be read side by side.")
    say("=" * 78)

    if args.report:
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[ground] wrote {args.report}")


if __name__ == "__main__":
    main()
