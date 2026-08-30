"""Regenerates every chart in the README from the committed JSON results.

No manual step and no plotting dependency: the SVG is written directly, so a clean
checkout can reproduce the charts with the standard library alone.

Colours are mid-tones chosen to read on both GitHub's light and dark themes, and the
background is left transparent for the same reason.

Usage: create_diagrams.py [results_dir] [output_dir]
"""

import json
import math
import pathlib
import sys

HAWK = "#2f7d8f"
TICK_LIKELIHOOD = "#b4622a"
TICK_LEASTSQ = "#8a7ab5"
AXIS = "#8b949e"
TEXT = "#8b949e"

W, H = 720, 380
LEFT, RIGHT, TOP, BOTTOM = 70, 170, 30, 52


def header(width=W, height=H):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" font-family="-apple-system,BlinkMacSystemFont,'
            f'Segoe UI,Helvetica,Arial,sans-serif" font-size="12">']


def legend(entries, x, y):
    out = []
    for i, (label, colour) in enumerate(entries):
        yy = y + i * 20
        out.append(f'<rect x="{x}" y="{yy - 9}" width="11" height="11" fill="{colour}" rx="2"/>')
        out.append(f'<text x="{x + 17}" y="{yy}" fill="{TEXT}">{label}</text>')
    return out


def log_line_chart(series, title, ylabel, path):
    """series: list of (label, colour, [(x, y), ...]) with x, y > 0. Log-log axes."""
    xs = [x for _, _, pts in series for x, _ in pts]
    ys = [y for _, _, pts in series for _, y in pts]
    x0, x1 = math.log10(min(xs)), math.log10(max(xs))
    y0, y1 = math.log10(min(ys)), math.log10(max(ys))
    y0, y1 = math.floor(y0), math.ceil(y1)
    pad = (x1 - x0) * 0.04 or 0.2

    def px(x): return LEFT + (math.log10(x) - x0 + pad) / (x1 - x0 + 2 * pad) * (W - LEFT - RIGHT)
    def py(y): return H - BOTTOM - (math.log10(y) - y0) / max(y1 - y0, 1e-9) * (H - TOP - BOTTOM)

    s = header()
    s.append(f'<text x="{LEFT}" y="18" fill="{TEXT}" font-size="13" font-weight="600">{title}</text>')
    for decade in range(y0, y1 + 1):
        y = py(10 ** decade)
        s.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{W - RIGHT}" y2="{y:.1f}" '
                 f'stroke="{AXIS}" stroke-opacity="0.25"/>')
        s.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" fill="{TEXT}" text-anchor="end">'
                 f'{fmt_decade(decade)}</text>')
    for x in sorted(set(xs)):
        s.append(f'<text x="{px(x):.1f}" y="{H - BOTTOM + 18:.1f}" fill="{TEXT}" '
                 f'text-anchor="middle">{human(x)}</text>')
    s.append(f'<line x1="{LEFT}" y1="{H - BOTTOM}" x2="{W - RIGHT}" y2="{H - BOTTOM}" stroke="{AXIS}"/>')
    s.append(f'<text x="{W / 2 - 40:.0f}" y="{H - 10}" fill="{TEXT}" text-anchor="middle">events</text>')
    s.append(f'<text x="16" y="{H / 2:.0f}" fill="{TEXT}" text-anchor="middle" '
             f'transform="rotate(-90 16 {H / 2:.0f})">{ylabel}</text>')
    for label, colour, pts in series:
        pts = sorted(pts)
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(pts))
        s.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2.2"/>')
        for x, y in pts:
            s.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4" fill="{colour}"/>')
    s += legend([(l, c) for l, c, _ in series], W - RIGHT + 14, TOP + 12)
    s.append("</svg>")
    path.write_text("\n".join(s) + "\n")
    print(f"wrote {path}", file=sys.stderr)


def grouped_bar_chart(groups, series_labels, colours, title, ylabel, path, note=None):
    """groups: list of (group_label, [value_or_None per series]). Log y axis."""
    values = [v for _, vs in groups for v in vs if v]
    y0, y1 = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
    def py(y): return H - BOTTOM - (math.log10(y) - y0) / max(y1 - y0, 1e-9) * (H - TOP - BOTTOM)

    s = header()
    s.append(f'<text x="{LEFT}" y="18" fill="{TEXT}" font-size="13" font-weight="600">{title}</text>')
    for decade in range(y0, y1 + 1):
        y = py(10 ** decade)
        s.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{W - RIGHT}" y2="{y:.1f}" '
                 f'stroke="{AXIS}" stroke-opacity="0.25"/>')
        s.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" fill="{TEXT}" text-anchor="end">'
                 f'{fmt_decade(decade)}</text>')
    span = (W - LEFT - RIGHT) / max(len(groups), 1)
    bar = span / (len(series_labels) + 1.4)
    for gi, (label, vs) in enumerate(groups):
        base = LEFT + gi * span + (span - bar * len(series_labels)) / 2
        for si, v in enumerate(vs):
            x = base + si * bar
            if v is None:
                s.append(f'<text x="{x + bar / 2:.1f}" y="{H - BOTTOM - 6}" fill="{TEXT}" '
                         f'text-anchor="middle" font-size="10">n/a</text>')
                continue
            top = py(v)
            s.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar - 3:.1f}" '
                     f'height="{H - BOTTOM - top:.1f}" fill="{colours[si]}" rx="2"/>')
        s.append(f'<text x="{LEFT + gi * span + span / 2:.1f}" y="{H - BOTTOM + 18:.1f}" '
                 f'fill="{TEXT}" text-anchor="middle">{label}</text>')
    s.append(f'<line x1="{LEFT}" y1="{H - BOTTOM}" x2="{W - RIGHT}" y2="{H - BOTTOM}" stroke="{AXIS}"/>')
    s.append(f'<text x="16" y="{H / 2:.0f}" fill="{TEXT}" text-anchor="middle" '
             f'transform="rotate(-90 16 {H / 2:.0f})">{ylabel}</text>')
    if note:
        s.append(f'<text x="{LEFT}" y="{H - 10}" fill="{TEXT}" font-size="10">{note}</text>')
    s += legend(list(zip(series_labels, colours)), W - RIGHT + 14, TOP + 12)
    s.append("</svg>")
    path.write_text("\n".join(s) + "\n")
    print(f"wrote {path}", file=sys.stderr)


def fmt_decade(d):
    return f"{10 ** d:g}s" if d >= 0 else f"{10 ** d:g}s".replace("0.", ".")


def human(x):
    for cut, suffix in ((1_000_000, "M"), (1_000, "k")):
        if x >= cut:
            return f"{x / cut:g}{suffix}"
    return f"{x:g}"


def hawk_runs(payload):
    """The per-point files hold one `run`; tolerate the aggregate `runs` shape too."""
    out = []
    for entry in payload if isinstance(payload, list) else [payload]:
        if "run" in entry:
            out.append(entry["run"])
        else:
            out.extend(entry.get("runs", []))
    return out


def tick_runs(payload):
    out = []
    for entry in payload if isinstance(payload, list) else [payload]:
        out.extend(entry.get("runs", []))
    return out


def main(results, out_dir):
    results, out_dir = pathlib.Path(results), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d1 = json.loads((results / "fit-d1.json").read_text())
    hawk = [(r["events"], r["seconds_median"]) for r in hawk_runs(d1["hawk"]) if r.get("completed")]
    tl = [(r["events"], r["seconds_median"]) for r in tick_runs(d1["tick"])
          if r.get("completed") and r["gofit"] == "likelihood"]
    ts = [(r["events"], r["seconds_median"]) for r in tick_runs(d1["tick"])
          if r.get("completed") and r["gofit"] == "least-squares"]
    log_line_chart(
        [("hawk, likelihood", HAWK, hawk),
         ("tick, likelihood", TICK_LIKELIHOOD, tl),
         ("tick, least-squares", TICK_LEASTSQ, ts)],
        "Univariate fit, wall clock (median of 5, log-log)", "seconds",
        out_dir / "fit-d1.svg")

    groups, missing = [], []
    for d in (1, 10, 100):
        path = results / f"fit-d{d}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        h = [r for r in hawk_runs(payload["hawk"]) if r.get("completed")]
        t = tick_runs(payload["tick"])
        tl_ = [r for r in t if r.get("completed") and r.get("gofit") == "likelihood"]
        ts_ = [r for r in t if r.get("completed") and r.get("gofit") == "least-squares"]
        biggest = max((r["nominal_n"] for r in h), default=None)
        if biggest is None:
            continue
        pick = lambda rs: next((r["seconds_median"] for r in rs if r["nominal_n"] == biggest), None)
        groups.append((f"d={d}\nn={human(biggest)}", [pick(h), pick(tl_), pick(ts_)]))
        if pick(tl_) is None:
            missing.append(f"d={d}")
    note = ("tick's likelihood objective does not run at " + ", ".join(missing) +
            " (benchmarks/README.md §5.4)") if missing else None
    grouped_bar_chart(
        groups, ["hawk, likelihood", "tick, likelihood", "tick, least-squares"],
        [HAWK, TICK_LIKELIHOOD, TICK_LEASTSQ],
        "Fit wall clock by dimension, largest completed n", "seconds",
        out_dir / "fit-by-dimension.svg", note)

    sim_path = results / "simulate.json"
    if sim_path.exists():
        sim = json.loads(sim_path.read_text())
        series = []
        for label, colour, side in (("hawk", HAWK, "hawk"), ("tick", TICK_LIKELIHOOD, "tick")):
            pts = []
            for entry in sim[side]:
                if entry["dimension"] != 1:
                    continue
                for r in entry["runs"]:
                    if r.get("completed"):
                        pts.append((r["events_median"], r["seconds_median"]))
            if pts:
                series.append((f"{label}, d=1", colour, pts))
        if series:
            log_line_chart(series, "Simulation to a fixed horizon, d=1 (median of 5, log-log)",
                           "seconds", out_dir / "simulate.svg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results",
         sys.argv[2] if len(sys.argv) > 2 else "docs/diagrams")
