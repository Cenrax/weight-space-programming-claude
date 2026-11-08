"""Exp2: can Claude tell what a tiny MLP DDPM generates from its weights alone (no tools, no forward passes)?

    venv/bin/python experiments/exp2/run.py --pilot                       # 1 call: main / circle / attempt 1
    venv/bin/python experiments/exp2/run.py --condition main --attempts 5
    venv/bin/python experiments/exp2/run.py --condition permuted --attempts 2
    venv/bin/python experiments/exp2/run.py --condition random --attempts 2
    venv/bin/python experiments/exp2/run.py --grade-only                  # regrade logs, figures, summary, blog data
    venv/bin/python experiments/exp2/run.py --dry-run --condition main    # print prompt size, no API call

Runs are idempotent: an attempt whose result.json is complete is skipped. Grading rules: experiments/exp2/GRADING.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "train"))

from harness.claude_client import ask, BudgetExhausted  # noqa: E402
from harness.logging_utils import write_result, read_result, new_run_id  # noqa: E402
from harness.weights_text import weights_to_text  # noqa: E402
from ddpm import build_model, sample  # noqa: E402
from eval.exp2_judge import score_answer, classify  # noqa: E402

EXP = "exp2"
LOGS = REPO / "logs" / EXP
CKPT = REPO / "checkpoints" / EXP
FIG = REPO / "docs" / "assets" / "figures"
SHAPES = ["circle", "square", "triangle"]
BUDGET_USD = 15.0
RESERVE_USD = 1.6          # worst case per call: ~73k in ($0.29) + 64k out ($1.28)
EFFORT = "high"
QUANT = 0.001
PERM_SEED = {"circle": 11, "square": 12, "triangle": 13}
RANDOM_SEEDS = [1001, 1002]

ALIASES = {
    "circle": ["circle", "circles", "circular", "disc", "discs", "disk", "disks", "round", "ring", "rings"],
    "square": ["square", "squares", "rectangle", "rectangles", "rectangular", "box", "boxes"],
    "triangle": ["triangle", "triangles", "triangular"],
}
UNTRAINED_WORDS = ["untrained", "not trained", "random", "noise", "noisy", "nothing", "no coherent", "incoherent",
                   "unstructured", "no structure", "no meaningful", "meaningless", "initialization", "initialized",
                   "initialised"]

# ----------------------------------------------------------------------------- prompt

PROMPT_HEAD = """Below are the complete weights of a small neural network, printed as text.

The network is the denoiser of a denoising diffusion model (DDPM) for 16x16 single-channel (grayscale) images.

Architecture (an MLP):
- Input: 272 numbers = concat(x_t, temb(t)).
  - Input units 0..255: x_t, the noisy image, flattened row-major: input index = row*16 + col (row 0 is the top row, col 0 is the left column). Clean pixel values lie in [-1, 1], where -1 = black and +1 = white.
  - Input units 256..271: temb(t), a fixed (not learned) sinusoidal embedding of the timestep t in {0, ..., 199}: temb(t)[k] = sin(t * f_k) for k = 0..7 and temb(t)[8 + k] = cos(t * f_k), with f_k = 1000^(-k/8).
- Layer 1: Linear(272 -> 48), then SiLU.
- Layer 2: Linear(48 -> 48), then SiLU.
- Layer 3: Linear(48 -> 256), no activation.
- Output: 256 numbers, the model's prediction of the clean image x0, in the same row-major 16x16 layout (output index = row*16 + col).

Diffusion: T = 200 steps with a cosine noise schedule; x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps with eps ~ N(0, I). The network was trained with an MSE loss to predict the clean image x0 from (x_t, t). Sampling starts from pure Gaussian noise at t = 199 and steps down to t = 0, each step using the predicted x0 (clipped to [-1, 1]) in the standard DDPM posterior; the final image is the last predicted x0.

Weight format: "W<k> row i" lists the weights from input unit i of layer k to every output unit of that layer (the matrix is printed as [in, out]); "b<k>" is the bias vector of layer k. All values are integers: the real value is integer * 0.001. W1 has 272 rows of 48, W2 has 48 rows of 48, W3 has 48 rows of 256.

"""

PROMPT_TAIL = """
Question: what images does this model generate?

No tools are allowed, and no forward passes: do not run or numerically simulate the network on any input. Answer by reading the weights. If you think the model does not generate anything coherent, say so.

Please give:
(a) a short description of the images this model generates;
(b) a drawing of one typical generated image as exactly 16 lines of 16 characters, '#' for a bright (white) pixel and '.' for a dark (black) pixel, in a code block;
(c) your confidence, from 0 to 1, that your description is correct.

End your reply with one final line in exactly this format (valid JSON):
ANSWER: {"description": "...", "confidence": 0.x}
"""


def build_prompt(sd) -> str:
    return PROMPT_HEAD + weights_to_text(sd, quantize=QUANT, activation="silu") + PROMPT_TAIL


# ----------------------------------------------------------------------------- networks

def load_sd(shape):
    return torch.load(CKPT / f"{shape}.pt", map_location="cpu")


def net_from_sd(sd):
    m = build_model()
    m.load_state_dict(sd)
    return m.eval()


def permute_sd(sd, seed):
    """Consistently permute hidden layer 1 (48) and hidden layer 2 (48). Returns (new_sd, max_abs_diff)."""
    g = torch.Generator().manual_seed(seed)
    p1, p2 = torch.randperm(48, generator=g), torch.randperm(48, generator=g)
    new = {k: v.clone() for k, v in sd.items()}
    new["0.weight"], new["0.bias"] = sd["0.weight"][p1], sd["0.bias"][p1]
    new["2.weight"], new["2.bias"] = sd["2.weight"][p2][:, p1], sd["2.bias"][p2]
    new["4.weight"] = sd["4.weight"][:, p2]
    x = torch.randn(256, 272, generator=g)
    with torch.no_grad():
        diff = (net_from_sd(sd)(x) - net_from_sd(new)(x)).abs().max().item()
    assert diff < 1e-4, diff
    assert not torch.equal(new["4.weight"], sd["4.weight"])
    return new, diff


def random_sd(seed):
    torch.manual_seed(seed)
    return build_model().state_dict()


def jobs_for(condition, attempts):
    """-> list of (condition, target, attempt, sd_factory, extra)."""
    out = []
    if condition == "main":
        for sh in SHAPES:
            for a in range(1, attempts + 1):
                out.append((condition, sh, a, lambda sh=sh: (load_sd(sh), {})))
    elif condition == "permuted":
        for sh in SHAPES:
            def f(sh=sh):
                sd, diff = permute_sd(load_sd(sh), PERM_SEED[sh])
                return sd, {"perm_seed": PERM_SEED[sh], "perm_max_abs_diff": diff}
            for a in range(1, attempts + 1):
                out.append((condition, sh, a, f))
    elif condition == "random":
        for a, seed in enumerate(RANDOM_SEEDS[:attempts], 1):
            out.append((condition, "untrained", a, lambda seed=seed: (random_sd(seed), {"init_seed": seed})))
    else:
        raise ValueError(condition)
    return out


# ----------------------------------------------------------------------------- parsing / grading

def parse_answer(text):
    desc, conf, ok = "", None, False
    lines = [l for l in text.splitlines() if l.strip().startswith("ANSWER:")]
    if lines:
        raw = lines[-1].strip()[len("ANSWER:"):].strip().strip("`")
        try:
            j = json.loads(raw)
            desc, conf, ok = str(j.get("description", "")), j.get("confidence"), True
        except json.JSONDecodeError:
            m = re.search(r'"description"\s*:\s*"([^"]*)"', raw)
            desc = m.group(1) if m else ""
            m = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
            conf = m.group(1) if m else None
    try:
        conf = None if conf is None else min(1.0, max(0.0, float(conf)))
    except ValueError:
        conf = None
    return desc, conf, ok


def parse_drawing(text):
    rows = [l.strip().strip("`").strip() for l in text.splitlines()]
    good = [bool(re.fullmatch(r"[#.]{16}", r)) for r in rows]
    last = None
    i = 0
    while i < len(rows):
        if good[i]:
            j = i
            while j < len(rows) and good[j]:
                j += 1
            if j - i == 16:
                last = rows[i:j]
            i = j
        else:
            i += 1
    return "\n".join(last) if last else None


def text_label(desc):
    first, mentioned = None, []
    pos = {}
    for cls, al in ALIASES.items():
        for a in al:
            m = re.search(rf"\b{re.escape(a)}\b", desc, re.I)
            if m:
                pos[cls] = min(pos.get(cls, 10 ** 9), m.start())
    mentioned = sorted(pos, key=pos.get)
    first = mentioned[0] if mentioned else "none"
    return first, mentioned


def says_untrained(desc):
    return any(re.search(rf"\b{re.escape(w)}\b", desc, re.I) for w in UNTRAINED_WORDS)


def grade(d: Path):
    r = read_result(d)
    text = (d / "response.md").read_text() if (d / "response.md").exists() else ""
    desc, conf, parsed = parse_answer(text)
    drawing = parse_drawing(text)
    if drawing:
        (d / "drawing.txt").write_text(drawing + "\n")
    label, mentioned = text_label(desc)
    target, cond = r.get("target"), r.get("condition")
    judge = None
    if drawing:
        judge = score_answer(drawing, target) if target in SHAPES else classify(drawing)
    if cond == "random":
        correct = says_untrained(desc) if parsed else False
        score = None
    else:
        correct = parsed and label == target
        score = int(bool(judge and judge.get("correct")))
    th = (d / "thinking.md").read_text() if (d / "thinking.md").exists() else ""
    excerpt = (th.strip() or text.strip())[:800]
    shape_name = target.capitalize() if target in SHAPES else "Random init"
    cname = {"main": "", "permuted": " (permuted)", "random": ""}[cond]
    write_result(d, answer=desc or "(no ANSWER line)", correct=bool(correct), confidence=conf, score=score,
                 label=f"{shape_name}{cname} · attempt {r.get('attempt')}",
                 group={"main": shape_name, "permuted": f"{shape_name} (permuted)",
                        "random": "Random-init control"}[cond],
                 reasoning_excerpt=excerpt, text_label=label, labels_mentioned=mentioned, answer_parsed=parsed,
                 drawing=drawing, drawing_judge=None if not judge else
                 {k: judge.get(k) for k in ("pred", "scores", "variant", "on_pixels")})
    return read_result(d)


# ----------------------------------------------------------------------------- running

def exp_spend():
    tot = 0.0
    for p in LOGS.glob("*/result.json"):
        try:
            tot += float(json.loads(p.read_text()).get("cost_usd") or 0)
        except (ValueError, OSError):
            pass
    return tot


def existing(condition, target, attempt):
    for p in LOGS.glob("*/result.json"):
        r = json.loads(p.read_text())
        if (r.get("condition"), r.get("target"), r.get("attempt")) == (condition, target, attempt) \
                and r.get("status") == "complete":
            return p.parent
    return None


def run_job(job):
    condition, target, attempt, factory = job
    if (d := existing(condition, target, attempt)):
        print(f"skip {condition}/{target}/{attempt}: {d.name}")
        return d
    spent = exp_spend()
    if spent + RESERVE_USD > BUDGET_USD:
        print(f"BUDGET: exp2 spend ${spent:.2f} + reserve ${RESERVE_USD} > ${BUDGET_USD}; not starting "
              f"{condition}/{target}/{attempt}")
        return None
    sd, extra = factory()
    prompt = build_prompt(sd)
    rid = new_run_id(f"{condition}-{target}-a{attempt}")
    print(f"start {rid}")
    r = ask(prompt, exp_id=EXP, run_id=rid, effort=EFFORT, max_tokens=64000, condition=condition,
            max_cost_usd=RESERVE_USD, extra={"target": target, "attempt": attempt, "effort": EFFORT,
                                             "quantize": QUANT, **extra})
    res = grade(r.log_dir)
    print(f"done  {rid}: ${r.cost_usd:.3f} in={res['tokens_in']} out={res['tokens_out']} stop={r.stop_reason} "
          f"label={res['text_label']} correct={res['correct']} draw={res.get('drawing_judge')} "
          f"conf={res['confidence']} | {res['answer'][:120]}")
    return r.log_dir


# ----------------------------------------------------------------------------- report

def all_results():
    out = []
    for p in sorted(LOGS.glob("*/result.json")):
        r = json.loads(p.read_text())
        r["_dir"] = p.parent
        out.append(r)
    return out


def grid_to_arr(s):
    return np.array([[1.0 if c == "#" else 0.0 for c in row] for row in s.splitlines()], dtype=np.float32)


def make_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    media = []
    for sh in SHAPES:
        runs = sorted([r for r in results if r.get("condition") == "main" and r.get("target") == sh],
                      key=lambda r: r.get("attempt") or 0)[:5]
        torch.manual_seed(0)
        real = sample(net_from_sd(load_sd(sh)), 8, seed=7).numpy()
        fig, axes = plt.subplots(2, 8, figsize=(8, 2.35), dpi=80)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#888")
        for i in range(8):
            ax = axes[0, i]
            if i < len(runs) and runs[i].get("drawing"):
                ax.imshow(grid_to_arr(runs[i]["drawing"]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                mark = "✓" if runs[i].get("correct") else "✗"
                ax.set_title(f"try {runs[i].get('attempt')} {mark}", fontsize=8)
            elif i < len(runs):
                ax.set_title(f"try {runs[i].get('attempt')}: none", fontsize=8)
            else:
                ax.axis("off")
            axes[1, i].imshow(real[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0, 0].set_ylabel("Claude", fontsize=9)
        axes[1, 0].set_ylabel("model", fontsize=9)
        fig.tight_layout(pad=0.3)
        out = FIG / f"exp2_claude_{sh}.png"
        fig.savefig(out, dpi=80)
        plt.close(fig)
        n_draw = sum(1 for r in runs if r.get("drawing"))
        n_fill = sum(1 for r in runs if (r.get("drawing_judge") or {}).get("variant") == "filled")
        media.append({"type": "png", "src": f"assets/figures/exp2_claude_{sh}.png", "group": sh.capitalize(),
                      "pixel": True,
                      "caption": f"{sh.capitalize()} DDPM. Top: Claude's {n_draw} drawings from the weights alone "
                                 f"(✓/✗ = text label graded correct). Bottom: 8 real samples from the model (seed 7).",
                      "alt": f"Top row: {n_draw} 16x16 drawings by Claude of what the {sh} model generates "
                             f"({n_fill} judged filled shapes, {n_draw - n_fill} outlines). Bottom row: eight real "
                             f"samples from the {sh} diffusion model, filled white {sh}s on black."})
        print("wrote", out, out.stat().st_size, "bytes")
    return media


def report(results):
    media = make_figures(results)
    rows, cats, s_txt, s_draw, s_perm = [], [], [], [], []
    tot_main = tot_main_ok = tot_draw_ok = 0
    for sh in SHAPES:
        m = [r for r in results if r.get("condition") == "main" and r.get("target") == sh]
        p = [r for r in results if r.get("condition") == "permuted" and r.get("target") == sh]
        ok = sum(bool(r.get("correct")) for r in m)
        dr = sum(int(r.get("score") or 0) for r in m)
        pk = sum(bool(r.get("correct")) for r in p)
        confs = [r["confidence"] for r in m if r.get("confidence") is not None]
        fills = [(r.get("drawing_judge") or {}).get("variant") for r in m]
        rows.append([sh.capitalize(), f"{ok}/{len(m)}", f"{dr}/{len(m)}", f"{pk}/{len(p)}",
                     f"{np.mean(confs):.2f}" if confs else "–",
                     f"{fills.count('filled')} filled / {fills.count('outline')} outline"])
        cats.append(sh.capitalize())
        s_txt.append(round(100 * ok / len(m)) if m else 0)
        s_draw.append(round(100 * dr / len(m)) if m else 0)
        s_perm.append(round(100 * pk / len(p)) if p else 0)
        tot_main += len(m); tot_main_ok += ok; tot_draw_ok += dr
    rnd = [r for r in results if r.get("condition") == "random"]
    rconfs = [r["confidence"] for r in rnd if r.get("confidence") is not None]
    rows.append(["Random init (control)", f"{sum(bool(r.get('correct')) for r in rnd)}/{len(rnd)} said untrained",
                 "–", "–", f"{np.mean(rconfs):.2f}" if rconfs else "–", "; ".join(
                     (r.get("answer") or "")[:60] for r in rnd)])
    perm = [r for r in results if r.get("condition") == "permuted"]
    cost = sum(r.get("cost_usd") or 0 for r in results)
    summary = (f"From the weights alone, with no tools or forward passes, Claude Opus 5.5 named the right shape in "
               f"{tot_main_ok} of {tot_main} attempts and drew a judge-recognised correct shape in {tot_draw_ok} of "
               f"{tot_main}; on hidden-unit-permuted copies it was right in "
               f"{sum(bool(r.get('correct')) for r in perm)} of {len(perm)}, and it called "
               f"{sum(bool(r.get('correct')) for r in rnd)} of {len(rnd)} untrained networks untrained "
               f"(total API cost ${cost:.2f}).")
    # Post-hoc caveat (does not change the pre-registered grades): flag label hits whose description leads with
    # "digits", and drawing hits whose judge margin over the runner-up class is < 0.1 IoU.
    digit_hits = [r for r in results if r.get("correct") and r.get("condition") != "random"
                  and re.search(r"\bdigit", r.get("answer") or "", re.I)]
    weak_draw = [r for r in results if r.get("condition") == "main" and r.get("score") == 1
                 and (lambda s: s[0] - s[1] < 0.1)(sorted((r.get("drawing_judge") or {}).get("scores", {}).values(),
                                                          reverse=True) + [0, 0])]
    if digit_hits or weak_draw:
        summary += (f" Caveat: {len(digit_hits)} label hit(s) come from descriptions that answer 'handwritten "
                    f"digits' and only mention a ring-like '0'; {len(weak_draw)} drawing hit(s) win the judge's "
                    f"argmax by less than 0.1 IoU (digit-like drawings, not shapes).")
    js = {
        "status": "done",
        "summary": summary,
        "table": {"caption": "Per shape: text label (primary), drawing judged by template IoU (secondary), "
                             "permutation control, mean confidence (main), and drawing style.",
                  "columns": ["Network", "Text label correct (main)", "Drawing correct (main)",
                              "Permuted: label correct", "Mean confidence", "Drawing style / answers"],
                  "rows": rows},
        "media": media,
        "charts": [{"type": "bar", "title": "Exp2: accuracy by shape", "subtitle":
                    "% of attempts correct; 5 main attempts and 2 permuted attempts per shape",
                    "unit": "% correct", "categories": cats,
                    "series": [{"name": "Text label (main)", "values": s_txt},
                               {"name": "Drawing (main)", "values": s_draw},
                               {"name": "Text label (permuted)", "values": s_perm}],
                    "caption": "Text label is the pre-registered primary metric (alias match on the description). "
                               "Drawing is scored by template IoU, with outlines counting."}],
        "cost_usd": round(cost, 4),
    }
    (LOGS / "summary.json").write_text(json.dumps(js, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: js[k] for k in ("summary", "table")}, indent=1, ensure_ascii=False))
    import subprocess
    subprocess.run([sys.executable, str(REPO / "scripts" / "build_blog_data.py"), "--exp", EXP], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true", help="one call: main / circle / attempt 1")
    ap.add_argument("--condition", choices=["main", "permuted", "random", "all"], default=None)
    ap.add_argument("--attempts", type=int, default=None)
    ap.add_argument("--grade-only", action="store_true", help="regrade every run in logs/exp2 and rebuild outputs")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and count tokens; no model call")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    if a.grade_only:
        res = [grade(r["_dir"]) | {"_dir": r["_dir"]} for r in all_results() if r.get("status") == "complete"]
        report(res)
        return
    if a.pilot:
        jobs = jobs_for("main", 5)[:1]
    else:
        conds = ["main", "permuted", "random"] if a.condition in (None, "all") else [a.condition]
        default_att = {"main": 5, "permuted": 2, "random": 2}
        jobs = [j for c in conds for j in jobs_for(c, a.attempts or default_att[c])]
    if a.dry_run:
        from harness.weights_text import count_tokens
        sd, extra = jobs[0][3]()
        p = build_prompt(sd)
        print(f"{len(jobs)} job(s); prompt chars={len(p)} tokens={count_tokens(p)} extra={extra}")
        print(p[:3000]); print("..."); print(p[-1500:])
        return
    print(f"{len(jobs)} job(s); exp2 spend so far ${exp_spend():.2f}")
    try:
        with ThreadPoolExecutor(max_workers=min(3, a.workers)) as ex:
            list(ex.map(run_job, jobs))
    except BudgetExhausted as e:
        print("PROJECT BUDGET EXHAUSTED:", e)
    print(f"exp2 spend now ${exp_spend():.2f}")


if __name__ == "__main__":
    main()
