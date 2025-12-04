"""
Compute win@p curves for authors (agents) from Trace JSONL files.

Definition (per request):
- For each workload group (same definition + workload.uuid), identify a baseline latency:
  * Prefer runs whose AUTHOR is 'flashinfer' (case-insensitive). If none, use the fastest run in that group.
- For an author A with latency L_A and baseline latency L_B in the same group, define r = L_B / L_A.
- win@p for author A is the fraction of groups where r > p.
- We pre-calc all possible p's as the union of all r's across all authors (i.e., p-grid).

Inputs:
- One or more JSONL Trace files. Each line is a Trace object, matching your schema.
- Optional solution->author map JSON file if author is not embedded in the 'solution' string.

Outputs:
- CSV with columns: author,p,win_ratio,n_total,n_wins
  (n_total is the number of workload groups where this author had a valid run; n_wins counts those with r>p)

Examples:
  python win_at_p.py traces/*/*.jsonl -o win_at_p.csv
  python win_at_p.py traces/*/*.jsonl -o win_at_p.csv --baseline-author torch
  python win_at_p.py traces/*/*.jsonl -o win_at_p.csv --author-map author_map.json
  # author_map.json example:
  # { "rmsnorm_triton_v1": "alice", "my_fast_impl": "bob" }

Notes:
- If multiple runs exist for the same author within a group, we take the MIN latency for that author in that group.
- By default, the baseline author ('flashinfer') is EXCLUDED from output curves; use --include-baseline to include it.
"""

import argparse
import csv
import glob
import io
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib

# ---------- Parsing helpers ----------


def load_author_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError("author map must be a JSON object of {solution_name: author}")
    return {str(k): str(v) for k, v in m.items()}


def infer_author_from_solution(solution: str) -> str:
    """
    Heuristics to extract an 'author' from solution string.
    If none match, return the solution itself (treat solution as author).
    """
    s = str(solution)
    if "_" in s:
        return s.split("_", 1)[0]
    return s


def get_author(solution: str, sol2author: Dict[str, str]) -> str:
    return sol2author.get(solution) or infer_author_from_solution(solution)


# ---------- Data model ----------


class Run:
    __slots__ = ("definition", "workload_uuid", "solution", "author", "latency_ms")

    def __init__(
        self, definition: str, workload_uuid: str, solution: str, author: str, latency_ms: float
    ):
        self.definition = definition
        self.workload_uuid = workload_uuid
        self.solution = solution
        self.author = author
        self.latency_ms = float(latency_ms) if latency_ms is not None else None


GroupKey = Tuple[str, str]  # (definition, workload_uuid)

# ---------- I/O ----------


def iter_trace_lines(paths: Iterable[str]) -> Iterable[dict]:
    for p in paths:
        with io.open(p, "r", encoding="utf-8") as f:
            for ln, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse JSON in {p}:{ln}: {e}")
                yield obj


def collect_runs(paths: Iterable[str], author_map: Dict[str, str]) -> List[Run]:
    runs: List[Run] = []
    for obj in iter_trace_lines(paths):
        try:
            definition = obj["definition"]
            solution = obj["solution"]
            workload = obj["workload"]
            uuid = workload["uuid"]
            evalo = obj["evaluation"]
            status = evalo["status"]
            perf = evalo.get("performance")
        except KeyError as e:
            # Skip malformed records
            sys.stderr.write(f"WARNING: skipping record missing key {e}\n")
            continue

        latency = None
        if status == "PASSED" and perf and perf.get("latency_ms") is not None:
            latency = perf["latency_ms"]

        author = get_author(solution, author_map)
        runs.append(Run(definition, uuid, solution, author, latency))
    return runs


# ---------- Grouping & baseline ----------


def group_runs_by_workload(runs: List[Run]) -> Dict[GroupKey, List[Run]]:
    groups: Dict[GroupKey, List[Run]] = defaultdict(list)
    for r in runs:
        groups[(r.definition, r.workload_uuid)].append(r)
    return groups


def select_min_latency_per_author(runs: List[Run]) -> Dict[str, Run]:
    """Within a group, keep the best (min latency) run per author."""
    best: Dict[str, Run] = {}
    for r in runs:
        if r.latency_ms is None:
            continue
        cur = best.get(r.author)
        if cur is None or r.latency_ms < cur.latency_ms:
            best[r.author] = r
    return best


def choose_baseline(best_by_author: Dict[str, Run], baseline_author: str) -> Tuple[str, float]:
    """
    Returns (baseline_author_effective, baseline_latency).
    Prefer provided baseline_author if present; else fall back to fastest run.
    """
    # case-insensitive lookup for baseline author
    bl_key = None
    baseline_author_ci = baseline_author.lower()
    for a in best_by_author:
        if a.lower() == baseline_author_ci:
            bl_key = a
            break

    if bl_key is not None:
        return (bl_key, best_by_author[bl_key].latency_ms)

    # Fallback to fastest author in this group
    fastest_author = min(best_by_author.items(), key=lambda kv: kv[1].latency_ms)[0]
    return (fastest_author, best_by_author[fastest_author].latency_ms)


# ---------- win@p computation ----------


def compute_ratios_by_author(
    groups: Dict[GroupKey, List[Run]], baseline_author: str
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    """
    For each author, compute list of r = baseline_latency / author_latency
    over all groups where BOTH baseline and author exist (after baseline fallback).
    Returns:
      ratios_by_author: author -> list of r
      totals_by_author: author -> #groups where author had a run (and a baseline existed)
    """
    ratios_by_author: Dict[str, List[float]] = defaultdict(list)
    totals_by_author: Dict[str, int] = defaultdict(int)

    for _gk, runs in groups.items():
        best_by_author = select_min_latency_per_author(runs)
        if not best_by_author:
            continue

        bl_author_eff, bl_lat = choose_baseline(best_by_author, baseline_author)

        for author in {r.author for r in runs}:  # all authors who attempted
            if author == bl_author_eff:
                # skip the baseline author itself here
                continue
            totals_by_author[author] += 1

            run = best_by_author.get(author)
            if run is None or bl_lat <= 0 or run.latency_ms is None or run.latency_ms <= 0:
                # author had no valid PASSED latency → always lose (no r recorded)
                continue
            r = bl_lat / run.latency_ms
            ratios_by_author[author].append(r)

    return ratios_by_author, totals_by_author


def build_p_grid(ratios_by_author: Dict[str, List[float]]) -> List[float]:
    """Union of all r's across authors, sorted ascending, unique."""
    s: Set[float] = set()
    for arr in ratios_by_author.values():
        s.update(arr)
    return sorted(s)


def win_curve_for_author(r_values: List[float], p_grid: List[float]) -> List[Tuple[float, int]]:
    """
    Given r-values for an author and the global p-grid, return list of (p, wins_at_p),
    where wins_at_p counts r > p (strict).
    Efficiently computed by sorting r-values once.
    """
    if not r_values:
        return [(p, 0) for p in p_grid]
    r_sorted = sorted(r_values)
    n = len(r_sorted)
    out: List[Tuple[float, int]] = []
    # Two-pointer sweep: for each p ascending, count how many r > p
    idx = 0  # first index with r > p will move forward as p increases
    # But since we need r > p (strict), we can binary search each p; do linear pass for simplicity.
    for p in p_grid:
        # advance idx while r_sorted[idx] <= p
        while idx < n and r_sorted[idx] <= p:
            idx += 1
        wins = n - idx
        out.append((p, wins))
    return out


# ---------- Baseline inclusion (optional) ----------


def add_baseline_author_curve(
    groups: Dict[GroupKey, List[Run]],
    baseline_author: str,
    ratios_by_author: Dict[str, List[float]],
    totals_by_author: Dict[str, int],
    include_name: str,
) -> None:
    """
    Optionally include an explicit curve for the baseline author itself.
    For baseline vs baseline, r = 1 for every group where a baseline existed.
    """
    count = 0
    for _gk, runs in groups.items():
        best_by_author = select_min_latency_per_author(runs)
        if not best_by_author:
            continue
        bl_author_eff, _bl_lat = choose_baseline(best_by_author, baseline_author)
        if bl_author_eff in best_by_author:
            count += 1
    totals_by_author[include_name] = count
    ratios_by_author[include_name] = [1.0] * count  # r = 1 each group


# ---------- CSV output ----------


def write_curves_csv(
    out_path: str,
    p_grid: List[float],
    ratios_by_author: Dict[str, List[float]],
    totals_by_author: Dict[str, int],
    authors_order: Optional[List[str]] = None,
) -> None:
    if authors_order is None:
        authors_order = sorted(ratios_by_author.keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["author", "p", "win_ratio", "n_total", "n_wins"])
        for author in authors_order:
            rvals = ratios_by_author.get(author, [])
            total = totals_by_author.get(author, 0)
            curve_counts = win_curve_for_author(rvals, p_grid)
            for p, wins in curve_counts:
                win_ratio = (wins / total) if total > 0 else 0.0
                w.writerow([author, f"{p:.9g}", f"{win_ratio:.9g}", total, wins])


def win_curve_ratio_for_author(
    r_values: List[float], p_grid: List[float], total: int
) -> List[Tuple[float, float, int]]:
    """
    Returns (p, win_ratio, n_wins) for each p in p_grid for a single author.
    """
    # total = len(r_values)
    counts = win_curve_for_author(r_values, p_grid)  # (p, n_wins)
    out: List[Tuple[float, float, int]] = []
    for p, n_wins in counts:
        win_ratio = (n_wins / total) if total > 0 else 0.0
        # print(f"p={p}, n_wins={n_wins}, total={total}, win_ratio={win_ratio}")
        out.append((p, win_ratio, n_wins))
    return out


def select_authors_for_plot(
    ratios_by_author: Dict[str, List[float]],
    totals_by_author: Dict[str, int],
    authors_order: List[str],
    include: Optional[List[str]],
    top_k: Optional[int],
    min_groups: int,
) -> List[str]:
    """
    Choose which authors to display on the plot.
    Priority:
      1) if include is provided, use that (filtered by availability & min_groups)
      2) else take top_k authors from authors_order (which is already performance-sorted)
    """
    # filter by min_groups
    eligible = [a for a in authors_order if totals_by_author.get(a, 0) >= min_groups]
    if include:
        wanted = []
        include_set = set(x.strip() for x in include if x and x.strip())
        for a in eligible:
            if a in include_set or a.lower() in include_set:
                wanted.append(a)
        return wanted
    if top_k is not None and top_k > 0:
        return eligible[:top_k]
    return eligible  # show all eligible by default


def make_win_at_p_plot(
    out_path: Optional[str],
    p_grid: List[float],
    ratios_by_author: Dict[str, List[float]],
    totals_by_author: Dict[str, int],
    authors_to_plot: List[str],
    title: Optional[str] = None,
    xmax: Optional[float] = None,
    legend_outside: bool = True,
) -> None:
    """
    Render a single Win@p plot. If out_path is provided, saves there; otherwise shows interactively.
    """
    import matplotlib.pyplot as plt  # lazy import

    if not p_grid or not authors_to_plot:
        raise RuntimeError("Nothing to plot: empty p-grid or no authors selected.")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot a staircase curve for each author
    # We keep the default style and colors (no explicit colors).
    for author in authors_to_plot:
        rvals = ratios_by_author.get(author, [])
        curve = win_curve_ratio_for_author(
            rvals, p_grid, totals_by_author.get(author, 0)
        )  # (p, ratio, n_wins)
        xs = [p for (p, _, _) in curve]
        ys = [ratio for (_, ratio, _) in curve]
        # step plot conveys the strict threshold r > p nicely
        ax.step(xs, ys, where="post", label=f"{author}")

    ax.set_xlabel("p  (speedup over baseline)", fontsize=14)
    ax.set_ylabel("win@p  (fraction of workloads with r > p)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    # title = "Win@p Curves by Author" if title is None else title
    # ax.set_title(title)
    # ax.set_ylim(0.0, 1.0)
    if xmax is not None:
        ax.set_xlim(0.0, xmax)
    else:
        # small headroom on the right
        ax.set_xlim(0.0, max(p_grid) * 1.02 if p_grid else 1.0)

    if legend_outside:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        ax.legend()
        fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Compute win@p curves for authors from Trace JSONL files."
    )
    ap.add_argument("inputs", nargs="+", help="Input JSONL files or globs (e.g., logs/*.jsonl)")
    ap.add_argument("-o", "--output", required=True, help="Output CSV path for win@p curves.")
    ap.add_argument(
        "--author-map", default=None, help="Optional JSON file mapping solution->author."
    )
    ap.add_argument(
        "--baseline-author",
        default="flashinfer",
        help="Baseline author name to prefer when present (default: flashinfer).",
    )
    ap.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include a curve for the baseline author itself.",
    )

    # ---- plotting options ----
    ap.add_argument(
        "--plot",
        metavar="FIG_PATH",
        default=None,
        help="If set, save a Win@p figure to this path (e.g., win_at_p.png or .pdf).",
    )
    ap.add_argument(
        "--plot-show",
        action="store_true",
        help="Show the plot interactively instead of saving. If both --plot and --plot-show "
        "are given, the figure is saved AND shown.",
    )
    ap.add_argument(
        "--plot-authors",
        nargs="+",
        default=None,
        help="Explicit list of authors to include on the figure (default: auto-select).",
    )
    ap.add_argument(
        "--plot-top",
        type=int,
        default=10,
        help="Max number of authors to show if --plot-authors not provided (default: 10).",
    )
    ap.add_argument(
        "--plot-min-groups",
        type=int,
        default=3,
        help="Only plot authors with at least this many comparable groups (default: 3).",
    )
    ap.add_argument("--plot-title", type=str, default=None, help="Optional figure title.")
    ap.add_argument(
        "--plot-xmax", type=float, default=None, help="Optional x-axis max for p (e.g., 1.5)."
    )
    ap.add_argument(
        "--plot-legend-inside",
        action="store_true",
        help="Place legend inside the axes (default: outside).",
    )
    args = ap.parse_args()

    # Expand globs
    paths: List[str] = []
    for pat in args.inputs:
        matched = sorted(glob.glob(pat))
        if not matched and os.path.isfile(pat):
            matched = [pat]
        if not matched:
            sys.stderr.write(f"WARNING: no files matched '{pat}'\n")
        paths.extend(matched)

    if not paths:
        sys.stderr.write("ERROR: No input files found.\n")
        sys.exit(2)

    sol2author = load_author_map(args.author_map)
    runs = collect_runs(paths, sol2author)
    if not runs:
        sys.stderr.write("ERROR: No valid PASSED runs with latency found.\n")
        sys.exit(2)

    groups = group_runs_by_workload(runs)
    ratios_by_author, totals_by_author = compute_ratios_by_author(groups, args.baseline_author)

    if args.include_baseline:
        add_baseline_author_curve(
            groups,
            baseline_author=args.baseline_author,
            ratios_by_author=ratios_by_author,
            totals_by_author=totals_by_author,
            include_name=args.baseline_author,
        )

    if not ratios_by_author:
        sys.stderr.write("ERROR: No comparable author runs found relative to baseline.\n")
        sys.exit(2)

    p_grid = build_p_grid(ratios_by_author)

    # Order authors by median r descending (optional, just to make tables nicer)
    def median(x: List[float]) -> float:
        if not x:
            return 0.0
        xs = sorted(x)
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])

    authors_order = sorted(
        ratios_by_author.keys(), key=lambda a: median(ratios_by_author[a]), reverse=True
    )

    write_curves_csv(args.output, p_grid, ratios_by_author, totals_by_author, authors_order)

    sys.stderr.write(
        f"Processed {len(paths)} files, {len(groups)} workload groups; "
        f"wrote curves for {len(ratios_by_author)} authors to {args.output}\n"
    )

    # ---- make a plot ----
    if args.plot or args.plot_show:
        authors_to_plot = select_authors_for_plot(
            ratios_by_author,
            totals_by_author,
            authors_order,
            include=args.plot_authors,
            top_k=args.plot_top,
            min_groups=args.plot_min_groups,
        )
        if not authors_to_plot:
            sys.stderr.write("WARNING: No authors selected for plotting after filters.\n")
        else:
            try:
                make_win_at_p_plot(
                    out_path=args.plot,
                    p_grid=p_grid,
                    ratios_by_author=ratios_by_author,
                    totals_by_author=totals_by_author,
                    authors_to_plot=authors_to_plot,
                    title=args.plot_title,
                    xmax=args.plot_xmax,
                    legend_outside=not args.plot_legend_inside,
                )
                if args.plot:
                    sys.stderr.write(f"Saved plot to {args.plot}\n")
            except Exception as e:
                sys.stderr.write(f"WARNING: Failed to render plot: {e}\n")

    sys.stderr.write(
        f"Processed {len(paths)} files, {len(groups)} workload groups; "
        f"wrote curves for {len(ratios_by_author)} authors to {args.output}\n"
    )


if __name__ == "__main__":
    main()
