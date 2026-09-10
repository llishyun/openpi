#!/usr/bin/env python3
"""Plot loss / grad_norm from an openpi train log (the sbatch .out or logs/train_<job>.log).

    python scripts/slurm/plot_log.py <log> [-o out.png] [--smooth N]
    ssh n58 cat /local-data/user-data/$USER/job_<id>/slurm/*.out | python scripts/slurm/plot_log.py - -o loss.png

Parses lines like "Step 100: grad_norm=1.0130, loss=0.0977, param_norm=1802.4".
Writes a PNG (two stacked panels, one axis each) and prints a short text summary.
"""
import argparse
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAT = re.compile(r"Step (\d+): grad_norm=([\d.eE+-]+), loss=([\d.eE+-]+)")

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"


def smooth(xs, n):
    if n <= 1 or len(xs) < n:
        return xs
    out, acc = [], 0.0
    for i, v in enumerate(xs):
        acc += v
        if i >= n:
            acc -= xs[i - n]
        out.append(acc / min(i + 1, n))
    return out


def panel(ax, steps, ys, title, n_smooth):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    if n_smooth > 1:
        ax.plot(steps, ys, color=SERIES, linewidth=1, alpha=0.3, solid_joinstyle="round")
        ys = smooth(ys, n_smooth)
    ax.plot(steps, ys, color=SERIES, linewidth=2, solid_joinstyle="round", solid_capstyle="round")
    ax.scatter([steps[-1]], [ys[-1]], s=64, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"{ys[-1]:.4f}", (steps[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("step", color=INK2, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="log file, or - for stdin")
    ap.add_argument("-o", "--out", default="train_curve.png")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average window (in logged points)")
    a = ap.parse_args()
    text = sys.stdin.read() if a.log == "-" else open(a.log, errors="replace").read()
    rows = sorted({int(m[1]): (float(m[2]), float(m[3])) for m in PAT.finditer(text)}.items())
    if not rows:
        sys.exit("no 'Step N: grad_norm=..., loss=...' lines found")
    steps = [r[0] for r in rows]
    grad = [r[1][0] for r in rows]
    loss = [r[1][1] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, facecolor=SURFACE)
    panel(ax1, steps, loss, "loss", a.smooth)
    panel(ax2, steps, grad, "grad_norm", a.smooth)
    ax1.set_xlabel("")
    fig.suptitle(f"openpi training  -  {len(rows)} logged points, step {steps[0]}..{steps[-1]}",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, facecolor=SURFACE)

    k = min(10, len(loss))
    print(f"steps logged : {len(rows)}  (last step {steps[-1]})")
    print(f"loss         : first {loss[0]:.4f}  last {loss[-1]:.4f}  min {min(loss):.4f}  "
          f"mean(first {k}) {sum(loss[:k])/k:.4f}  mean(last {k}) {sum(loss[-k:])/k:.4f}")
    print(f"grad_norm    : first {grad[0]:.3f}  last {grad[-1]:.3f}  max {max(grad):.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
