import argparse
import json
from pathlib import Path

from flashinfer_bench.db.database import Database


def list_definitions(db: Database):
    for d in db.definitions:
        print(f"- {d.name}: {d.type} — {d.description}")


def list_solutions(db: Database):
    for s in db.solutions:
        print(f"- {s.name} (for {s.definition}) by {s.author}")


def list_traces(db: Database, definition: str = None):
    traces = db.traces if not definition else db.get_traces_for_definition(definition)
    for t in traces:
        print(f"- {t.definition} → {t.solution} [{t.evaluation.status}] "
              f"{t.evaluation.performance.speedup_factor:.2f}×")


def show_best(db: Database, definition: str, max_abs: float, max_rel: float):
    trace = db.get_best_op(definition, max_abs_diff=max_abs, max_relative_diff=max_rel)
    if not trace:
        print("No valid solution found for this workload.")
        return
    print(f"Best solution for {definition}:")
    print(f"- Solution: {trace.solution}")
    print(f"- Speedup:  {trace.evaluation.performance.speedup_factor:.2f}×")
    print(f"- Errors:   abs={trace.evaluation.correctness.max_absolute_error:.2e}, "
          f"rel={trace.evaluation.correctness.max_relative_error:.2e}")
    print(f"- Log:      {trace.evaluation.log_file}")


def main():
    parser = argparse.ArgumentParser(description="FlashInfer Bench CLI")
    parser.add_argument("db_path", type=str, help="Path to benchmark database folder")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-defs", help="List all workload definitions")
    subparsers.add_parser("list-sols", help="List all registered solutions")

    parser_list_traces = subparsers.add_parser("list-traces", help="List benchmark traces")
    parser_list_traces.add_argument("--definition", help="Filter by definition name")

    parser_best = subparsers.add_parser("best", help="Find best solution for a definition")
    parser_best.add_argument("definition", help="Definition name")
    parser_best.add_argument("--max-abs", type=float, default=1e-5, help="Max absolute error")
    parser_best.add_argument("--max-rel", type=float, default=1e-5, help="Max relative error")

    args = parser.parse_args()
    db = Database.from_uri(args.db_path)

    if args.command == "list-defs":
        list_definitions(db)
    elif args.command == "list-sols":
        list_solutions(db)
    elif args.command == "list-traces":
        list_traces(db, definition=args.definition)
    elif args.command == "best":
        show_best(db, args.definition, args.max_abs, args.max_rel)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
