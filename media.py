"""Sample videos and stills: original vs 2-step policy on the same task and start state.

    modal run modal_app.py --stage replay --plan main --items ref:2:42,steps2:2:42,...
    modal volume get monitor-under-compression media/main data/media
    python media.py                      # -> media/*.mp4, *.gif, *.png

Frames come from replaying each episode's recorded actions (modal_app.replay), so they show
the episodes behind the paper's numbers. The trace is the action-consistency monitor's
running score, divided by the threshold tuned on the original policy's successful runs
(alarm above 1.0, after a 10-step warm-up), as in the paper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import monitors as M

ROOT = Path(__file__).resolve().parent
DATA, FRAMES, OUT = ROOT / "data" / "main", ROOT / "data" / "media" / "main", ROOT / "media"
SUITE = "libero_10"
COLORS = {"ref": "#4a4a4a", "steps2": "#3f9e6e"}
ALARM, OK = "#d62728", "#2e7d32"
LABEL = {"ref": "Original policy (10 flow steps)", "steps2": "Sped-up policy (2 flow steps, 2.5× faster)"}
FPS, HOLD_S = 30, 2.0

# (kind, task, init): pairs chosen with scratch analysis of the test split, on tasks whose
# first-step features match across conditions (not 0, 1, 7: MuJoCo>=3.4 moves their props).
CLIPS = [("fail", 2, 42), ("fail", 4, 45), ("success", 2, 46)]
CAPTION = {
    "fail": "Both policies fail. The monitor raises an alarm for the original policy but never "
            "fires for the faster one:\nits successive plans agree more, so its score stays "
            "below the threshold tuned on the original.",
    "success": "Both policies succeed and neither alarms. The faster policy's score sits lower "
               "throughout, which is\nwhy it no longer reaches the old threshold when it fails.",
}


def _chunk(cond: str, task: int, init: int) -> np.ndarray:
    a = init // 10 * 10
    z = np.load(DATA / cond / SUITE / f"task{task}_init{a}-{a + 9}_s0.npz")
    b = list(z["init_ids"]).index(init)
    return z["chunk"][b, : int(z["length"][b])]


def threshold() -> float:
    """The paper's act_cons threshold: conformal over the original's successful calib runs."""
    scores = []
    for f in sorted((DATA / "ref" / SUITE).glob("task*_init30-39_s0.npz")):
        z = np.load(f)
        for b in range(len(z["init_ids"])):
            if z["success"][b]:
                L = int(z["length"][b])
                scores.append(M.episode_score(M.chunk_disagreement(z["chunk"][b, :L])))
    return M.conformal_tau(scores)


def load_clip(task: int, init: int, tau: float) -> dict:
    clip = {}
    for c in ("ref", "steps2"):
        z = np.load(FRAMES / f"{c}_task{task}_init{init}.npz")
        S = M.running_mean(M.chunk_disagreement(_chunk(c, task, init))) / tau
        alarm = M.first_alarm(M.chunk_disagreement(_chunk(c, task, init)), tau)
        clip[c] = {"agent": z["image"], "wrist": z["image2"], "S": S, "alarm": alarm,
                   "success": bool(z["success"]), "length": int(z["length"])}
        clip["task"] = str(z["task"])
        assert len(S) == clip[c]["length"], (c, len(S), clip[c]["length"])
    return clip


class Panel:
    """One 1280x720 frame: both policies side by side with their monitor traces."""

    def __init__(self, clip: dict, kind: str):
        self.clip, self.kind = clip, kind
        self.T = max(clip[c]["length"] for c in ("ref", "steps2"))
        ymax = max(1.5, 1.1 * max(clip[c]["S"][M.BURN_IN:].max() for c in ("ref", "steps2")))
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
        f = self.fig
        f.text(0.5, 0.965, f"“{clip['task']}”", ha="center", va="top", fontsize=17, weight="bold")
        f.text(0.5, 0.918, "Same task, same start state  •  SmolVLA on LIBERO-Long  •  "
               "failure monitor: action consistency, threshold tuned on the original policy",
               ha="center", va="top", fontsize=11, color="#555555")
        f.text(0.5, 0.025, CAPTION[kind], ha="center", va="bottom", fontsize=11, color="#333333")
        self.art = {}
        for j, c in enumerate(("ref", "steps2")):
            cx = 0.25 + j * 0.5  # column centre
            h = 0.46
            w = h * 720 / 1280  # square image in pixels
            left, bottom = cx - w / 2, 0.38
            f.text(cx, bottom + h + 0.012, LABEL[c], ha="center", va="bottom", fontsize=13, weight="bold",
                   color=COLORS[c])
            ax = f.add_axes([left, bottom, w, h])
            ax.set_axis_off()
            im = ax.imshow(clip[c]["agent"][0])
            border = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, lw=5, ec="none", clip_on=False)
            ax.add_patch(border)
            badge = ax.text(0.03, 0.04, "", transform=ax.transAxes, fontsize=13, weight="bold", color="white",
                            va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="none"))
            wi = 0.075
            hi = wi * 1280 / 720
            iax = f.add_axes([left + w - wi - 0.004, bottom + h - hi - 0.007, wi, hi])
            iax.set_xticks([]), iax.set_yticks([])
            for s in iax.spines.values():
                s.set_edgecolor("white"), s.set_linewidth(2)
            wim = iax.imshow(clip[c]["wrist"][0])
            sax = f.add_axes([cx - 0.19, 0.15, 0.38, 0.18])
            sax.axhline(1.0, color="k", ls="--", lw=1.2)
            sax.text(self.T, 1.0, "alarm threshold ", ha="right", va="bottom", fontsize=9)
            sax.axvspan(0, M.BURN_IN, color="#eeeeee", lw=0)
            sax.set_xlim(0, self.T), sax.set_ylim(0, ymax)
            sax.set_xlabel("timestep (20 Hz)", fontsize=10)
            sax.set_ylabel("score / threshold", fontsize=10)
            sax.spines[["top", "right"]].set_visible(False)
            sax.tick_params(labelsize=9)
            (line,) = sax.plot([], [], color=COLORS[c], lw=2)
            vline = sax.axvline(-10, color=ALARM, lw=1.5)
            note = sax.text(0.01, 0.97, "", transform=sax.transAxes, fontsize=11, weight="bold", va="top")
            self.art[c] = dict(im=im, wim=wim, border=border, badge=badge, line=line, vline=vline, note=note)

    def draw(self, t: int):
        for c in ("ref", "steps2"):
            d, a = self.clip[c], self.art[c]
            k = min(t, d["length"])  # frame k = after k env steps; frame 0 = start state
            a["im"].set_data(d["agent"][k])
            a["wim"].set_data(d["wrist"][k])
            s = min(k, len(d["S"]) - 1)
            a["line"].set_data(np.arange(s + 1), d["S"][: s + 1])
            # Stateless: every artist is set from t alone, so draw(t) can be called in any order.
            alarmed = d["alarm"] is not None and s >= d["alarm"]
            a["vline"].set_xdata([d["alarm"]] * 2 if alarmed else [-10, -10])
            a["border"].set_edgecolor(ALARM if alarmed else "none")
            if alarmed:
                a["note"].set_text(f"ALARM at step {d['alarm']}")
                a["note"].set_color(ALARM)
            elif t >= d["length"]:
                peak = d["S"][M.BURN_IN:].max()
                a["note"].set_text(f"no alarm (peak {peak:.2f}× threshold)")
                a["note"].set_color(COLORS[c])
            else:
                a["note"].set_text("")
            done = t >= d["length"]
            a["badge"].set_text(("SUCCESS" if d["success"] else "FAILED (time limit)") if done else "")
            a["badge"].get_bbox_patch().set_facecolor((OK if d["success"] else ALARM) if done else "none")

    def rgb(self) -> np.ndarray:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3]


def write_video(panel: Panel, mp4: Path, gif: Path):
    w, h = panel.fig.canvas.get_width_height()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    for t in list(range(panel.T + 1)) + [panel.T] * int(HOLD_S * FPS):
        panel.draw(t)
        ff.stdin.write(panel.rgb().tobytes())
    ff.stdin.close()
    assert ff.wait() == 0, mp4
    # GIF for web pages: 2x speed, 640 px wide, own palette.
    vf = "setpts=0.5*PTS,fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, str(gif)], check=True)


def filmstrip(clip: dict, kind: str, path: Path, steps: list[int]):
    """Poster still: keyframes for both policies over their monitor traces."""
    n = len(steps)
    fig = plt.figure(figsize=(2.0 * n + 1.2, 6.2), dpi=200, facecolor="white")
    fig.suptitle(f"“{clip['task']}”  (same start state)", fontsize=12, weight="bold", y=0.985)
    for r, c in enumerate(("ref", "steps2")):
        d = clip[c]
        fig.text(0.012, 0.79 - r * 0.29, LABEL[c].replace(" (", "\n("), rotation=90, ha="left", va="center",
                 fontsize=9, weight="bold", color=COLORS[c])
        for j, t in enumerate(steps):
            ax = fig.add_axes([0.06 + j * (0.93 / n), 0.66 - r * 0.29, 0.93 / n - 0.008, 0.27])
            k = min(t, d["length"])
            ax.imshow(d["agent"][k]), ax.set_xticks([]), ax.set_yticks([])
            alarmed = d["alarm"] is not None and k >= d["alarm"]
            for s in ax.spines.values():
                s.set_edgecolor(ALARM if alarmed else "#bbbbbb"), s.set_linewidth(3 if alarmed else 0.8)
            if r == 0:
                ax.set_title(f"step {t}", fontsize=9)
    sax = fig.add_axes([0.08, 0.075, 0.89, 0.26])
    T = max(clip[c]["length"] for c in ("ref", "steps2"))
    for c in ("ref", "steps2"):
        d = clip[c]
        sax.plot(d["S"], color=COLORS[c], lw=1.6, label=LABEL[c])
        if d["alarm"] is not None:
            sax.plot(d["alarm"], d["S"][d["alarm"]], "o", color=ALARM, ms=6, zorder=5)
            sax.annotate(f"alarm, step {d['alarm']}", (d["alarm"], d["S"][d["alarm"]]), xytext=(8, 6),
                         textcoords="offset points", color=ALARM, fontsize=8, weight="bold")
    sax.axhline(1.0, color="k", ls="--", lw=0.9)
    sax.text(T, 1.0, "threshold (tuned on original) ", ha="right", va="bottom", fontsize=7.5)
    sax.axvspan(0, M.BURN_IN, color="#eeeeee", lw=0)
    for t in steps:
        sax.axvline(t, color="#cccccc", lw=0.6, zorder=0)
    sax.set_xlim(0, T), sax.set_ylim(0, None)
    sax.set_xlabel("timestep (20 Hz)", fontsize=8), sax.set_ylabel("monitor score /\nthreshold", fontsize=8)
    sax.tick_params(labelsize=7), sax.spines[["top", "right"]].set_visible(False)
    sax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.18))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    tau = threshold()
    print(f"act_cons threshold {tau:.5f}")
    for kind, task, init in CLIPS:
        clip = load_clip(task, init, tau)
        stem = OUT / f"{kind}_task{task}_init{init}"
        for c in ("ref", "steps2"):
            d = clip[c]
            print(f"{stem.name} {c}: success={d['success']} length={d['length']} alarm={d['alarm']} "
                  f"peak={d['S'][M.BURN_IN:].max():.2f}x")
        panel = Panel(clip, kind)
        write_video(panel, stem.with_suffix(".mp4"), stem.with_suffix(".gif"))
        alarms = [clip[c]["alarm"] for c in ("ref", "steps2") if clip[c]["alarm"] is not None]
        t_still = min(alarms[0] + 40, panel.T) if alarms else panel.T
        panel.draw(t_still)
        panel.fig.savefig(stem.parent / f"{stem.name}_still.png", dpi=150)
        plt.close(panel.fig)
        T = panel.T
        keys = sorted({0, *alarms, T // 3, 2 * T // 3, T})
        filmstrip(clip, kind, stem.parent / f"{stem.name}_filmstrip.png", keys[:6])


if __name__ == "__main__":
    main()
