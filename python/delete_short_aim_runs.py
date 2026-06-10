#!/usr/bin/env python3
import argparse
import sys


MAX_DURATION_SECONDS = 30 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete Aim runs whose duration is less than 30 minutes.",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to the directory containing the .aim repository.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete matching runs. Without this flag, matching runs are only listed.",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(int(seconds), 60)
    return f"{minutes}m {remaining_seconds}s"


def matching_run_hashes(repo) -> tuple[list[str], int]:
    run_hashes = []
    skipped_count = 0
    for run in repo.iter_runs():
        if run.duration >= MAX_DURATION_SECONDS:
            continue
        if run.active:
            print(f"{run.hash} {format_duration(run.duration)} active skipped", flush=True)
            skipped_count += 1
            continue

        print(f"{run.hash} {format_duration(run.duration)} finished", flush=True)
        run_hashes.append(run.hash)
    return run_hashes, skipped_count


def main() -> int:
    args = parse_args()
    from aim import Repo

    repo = Repo.from_path(args.repo)
    run_hashes, skipped_count = matching_run_hashes(repo)
    if not run_hashes:
        print(
            f"No finished Aim runs shorter than 30 minutes found. "
            f"Skipped {skipped_count} active run(s)."
        )
        return 0
    if not args.yes:
        print(
            f"Would delete {len(run_hashes)} Aim run(s). "
            f"Skipped {skipped_count} active run(s). "
            "Run again with --yes to delete them."
        )
        return 0

    deleted, remaining = repo.delete_runs(run_hashes)
    if not deleted:
        print(
            f"Deleted {len(run_hashes) - len(remaining)} Aim run(s); "
            f"{len(remaining)} failed: {', '.join(remaining)}",
            file=sys.stderr,
        )
        return 1

    print(f"Deleted {len(run_hashes)} Aim run(s). Skipped {skipped_count} active run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
