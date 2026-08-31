"""Prints the README's benchmark tables from the committed JSON.

The README holds the tables as ordinary Markdown so it stays editable, and this script
is how they are produced and re-checked: run it and diff against the README. Every
number in the README's benchmark sections comes from here, and therefore from
`benchmarks/results/`.

Usage: readme_tables.py [results_dir]
"""

import json
import pathlib
import sys


def hawk_runs(payload):
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


def pick(runs, n, gofit=None):
    for r in runs:
        if r["nominal_n"] == n and (gofit is None or r.get("gofit") == gofit):
            return r
    return None


def secs(r):
    return f"{r['seconds_median']:.4f}" if r and r.get("completed") else "—"


def spread(r):
    return f"[{r['seconds_min']:.4f}, {r['seconds_max']:.4f}]" if r and r.get("completed") else "—"


def load(results, name):
    path = results / name
    return json.loads(path.read_text()) if path.exists() else None


def fit_d1(results):
    j = load(results, "fit-d1.json")
    if not j:
        return
    H, T = hawk_runs(j["hawk"]), tick_runs(j["tick"])
    print("`benchmarks/results/fit-d1.json`. Seconds; ratio is `hawk / tick`.\n")
    print("| events | hawk | hawk [min, max] | tick, likelihood | ratio | tick, least-squares | ratio |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for h in sorted([r for r in H if r.get("completed")], key=lambda r: r["nominal_n"]):
        n = h["nominal_n"]
        tl, ts = pick(T, n, "likelihood"), pick(T, n, "least-squares")
        rl = f"{h['seconds_median'] / tl['seconds_median']:.2f}x" if tl and tl.get("completed") else "—"
        rs = f"{h['seconds_median'] / ts['seconds_median']:.2f}x" if ts and ts.get("completed") else "—"
        print(f"| {h['events']:,} | {secs(h)} | {spread(h)} | {secs(tl)} | {rl} | {secs(ts)} | {rs} |")
    print("\nBoth answers under `hawk`'s unpenalized negative log-likelihood, "
          "so two objectives sit in one unit. Lower is better; the last two columns are "
          "how much worse `tick`'s answer scores.\n")
    print("| events | hawk nll | tick, likelihood | tick, least-squares |")
    print("| --- | --- | --- | --- |")
    for h in sorted([r for r in H if r.get("completed")], key=lambda r: r["nominal_n"]):
        n = h["nominal_n"]
        row = [f"| {h['events']:,} | {h['negative_log_likelihood']:.4f} "]
        for gofit in ("likelihood", "least-squares"):
            t = pick(T, n, gofit)
            v = t.get("negative_log_likelihood_under_hawk_objective") if t and t.get("completed") else None
            row.append(f"| +{v - h['negative_log_likelihood']:.4f} " if v is not None else "| — ")
        print("".join(row) + "|")


def fit_by_dimension(results):
    print("`benchmarks/results/fit-d{1,10,100}.json`. The largest `n` each dimension "
          "completed, seconds.\n")
    print("| d | parameters | events | hawk | tick, likelihood | tick, least-squares |")
    print("| --- | --- | --- | --- | --- | --- |")
    notes = []
    for d in (1, 10, 100):
        j = load(results, f"fit-d{d}.json")
        if not j:
            continue
        H, T = hawk_runs(j["hawk"]), tick_runs(j["tick"])
        done = [r for r in H if r.get("completed")]
        if not done:
            for r in H:
                notes.append(f"- `d = {d}`, `n = {r['nominal_n']:,}`: hawk **{r.get('abort_reason')}**")
            continue
        h = max(done, key=lambda r: r["nominal_n"])
        n = h["nominal_n"]
        tl, ts = pick(T, n, "likelihood"), pick(T, n, "least-squares")
        tl_cell = secs(tl) if tl and tl.get("completed") else "does not run"
        print(f"| {d} | {d + d * d + 1:,} | {h['events']:,} | {secs(h)} | {tl_cell} | {secs(ts)} |")
        for r in H:
            if not r.get("completed"):
                notes.append(f"- `d = {d}`, `n = {r['nominal_n']:,}`: hawk **{r.get('abort_reason')}**")
        for r in T:
            if not r.get("completed") and r.get("gofit") == "least-squares":
                notes.append(f"- `d = {d}`, `n = {r['nominal_n']:,}`: tick least-squares "
                             f"**{str(r.get('abort_reason'))[:110]}**")
    if notes:
        print("\nNot completed:\n")
        for line in dict.fromkeys(notes):
            print(line)


def simulate(results):
    j = load(results, "simulate.json")
    if not j:
        return
    print("`benchmarks/results/simulate.json`. One realization to a fixed horizon. The "
          "two use different generators, so the realized counts differ; both are shown.\n")
    print("| d | hawk events | hawk | tick events | tick | tick / hawk |")
    print("| --- | --- | --- | --- | --- | --- |")
    for he, te in zip(j["hawk"], j["tick"]):
        for hr, tr in zip(he["runs"], te["runs"]):
            if hr.get("completed") and tr.get("completed"):
                ratio = tr["seconds_median"] / hr["seconds_median"]
                print(f"| {he['dimension']} | {hr['events_median']:,} | {hr['seconds_median']:.4f} "
                      f"| {tr['events_median']:,} | {tr['seconds_median']:.4f} | {ratio:.2f}x |")


def window(results):
    j = load(results, "window-bias.json")
    if not j:
        return
    print(f"One realization of {j['events']:,} events, true baseline "
          f"`{j['true_baseline']}`. Only the declared window changes.\n")
    print("| dead time | declared horizon | hawk baseline | tick baseline |")
    print("| --- | --- | --- | --- |")
    for r in j["rows"]:
        print(f"| {r['dead_time_fraction']:.0%} | {r['declared_horizon']:,.0f} "
              f"| {r['hawk_baseline']:.4f} | {r['tick_baseline']:.4f} |")


def main(results):
    results = pathlib.Path(results)
    for title, fn in (("FIT_D1", fit_d1), ("FIT_DIM", fit_by_dimension),
                      ("SIMULATE", simulate), ("WINDOW", window)):
        print(f"\n<<<{title}>>>")
        fn(results)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results")
