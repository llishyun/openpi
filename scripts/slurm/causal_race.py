"""Submit held Slurm alternatives; the first preflight-ready job wins exactly once."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_CANDIDATES = (
    "A100-80GB/hpgpu/8",
    "A100-80GB/hpgpu/4",
    "A6000,RTX6000ADA,L40S/normal/8",
    "A6000,RTX6000ADA,L40S/normal/4",
)


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def pending_state(job):
    rows = command("squeue", "--noheader", "--user", str(os.getuid()), "--format=%i %T")
    return next((row.split()[1] for row in rows.splitlines() if row.split()[0] == job), "")


def submit(args):
    project = Path(__file__).resolve().parents[2]
    os.chdir(project)
    base = Path(os.environ["CHECKPOINT_BASE_DIR"]).expanduser().resolve()
    experiment = os.environ["EXP_NAME"]
    if Path(experiment).name != experiment or experiment in ("", ".", ".."):
        raise ValueError("EXP_NAME must be a single directory name.")
    run = base / "pi05_franka_pnp_overfit1_causal" / experiment
    if run.exists() or run.with_name(f"{experiment}_inference").exists():
        raise FileExistsError(f"Use a new EXP_NAME for a race: {run}")
    if args.replace_pending_job and pending_state(args.replace_pending_job) != "PENDING":
        raise ValueError("The old job is not PENDING; it was not changed.")
    race = base / ".causal_races" / experiment
    race.mkdir(parents=True, exist_ok=False)
    jobs = []
    releasing = False
    try:
        for candidate in args.candidate or DEFAULT_CANDIDATES:
            partition, qos, gpu = candidate.split("/")
            if gpu not in ("4", "8"):
                raise ValueError("Candidate GPU count must be 4 or 8.")
            env = dict(
                os.environ,
                CHECKPOINT_BASE_DIR=str(base),
                EXP_NAME=experiment,
                GPU_COUNT=gpu,
                CAUSAL_RACE_DIR=str(race),
                SMOKE_TEST="0",
                RESUME="0",
                PREFLIGHT_ONLY="0",
            )
            result = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    "--hold",
                    "--export=ALL",
                    f"--partition={partition}",
                    f"--qos={qos}",
                    f"--gres=gpu:{gpu}",
                    f"--time={args.time}",
                    f"--job-name=pi05_race_{gpu}g",
                    "scripts/slurm/franka_causal.sbatch",
                ],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                print(f"Skipped {candidate}: {result.stderr.strip()}", file=sys.stderr)
                continue
            # A single-cluster, non-array job is required for exact sibling cancellation.
            job = result.stdout.strip().split(";")[0]
            if not job.isdecimal():
                raise ValueError(f"Unexpected sbatch response: {result.stdout!r}")
            jobs.append(job)
            (race / "jobs.json").write_text(json.dumps(jobs) + "\n")
            print(f"Held {job}: {candidate}", flush=True)
        if not jobs:
            raise RuntimeError("No candidate accepted; existing job was not changed.")
        # Old job has no race lock. Only cancel it if it is still pending, before release.
        if args.replace_pending_job:
            if pending_state(args.replace_pending_job) != "PENDING":
                raise RuntimeError("Old job started/changed; cancelling new held candidates instead.")
            command("scancel", "--state=PENDING", args.replace_pending_job)
            if pending_state(args.replace_pending_job):
                raise RuntimeError("Old job still listed; new candidates will not be released.")
        releasing = True
        command("scontrol", "release", *jobs)
    except BaseException:
        if jobs and not releasing:
            subprocess.run(["scancel", *jobs], check=False)
        elif jobs:
            print(
                f"Release interrupted; inspect jobs {','.join(jobs)}. A released winner was not cancelled.",
                file=sys.stderr,
            )
        (race / "submission_failed").touch()
        raise
    print(f"Released jobs: {','.join(jobs)}\nRace directory: {race}", flush=True)
    print(f"Monitor: squeue --jobs={','.join(jobs)} --start --format='%.18i %.10T %.25S %.20Y %R'")


def claim(race, job):
    jobs = json.loads((race / "jobs.json").read_text())
    if not jobs or any(not isinstance(j, str) or not j.isdecimal() for j in jobs) or job not in jobs:
        raise ValueError("Job is not a member of this race.")
    # O_EXCL is atomic on the shared filesystem. The marker remains after exit:
    # a failed/requeued winner must never let another contender overwrite the run.
    try:
        with (race / "winner.json").open("x") as target:
            json.dump(
                {
                    "job_id": job,
                    "gpu_count": os.environ.get("GPU_COUNT"),
                    "partition": os.environ.get("SLURM_JOB_PARTITION"),
                },
                target,
            )
    except FileExistsError:
        print(f"Another job already claimed {race}; exiting without training.", flush=True)
        return 10
    siblings = [j for j in jobs if j != job]
    if siblings:
        result = subprocess.run(["scancel", *siblings], check=False)
        if result.returncode:
            print(
                "Some sibling cancellations failed; the shared winner marker still prevents duplicate training.",
                file=sys.stderr,
            )
    print(f"Winner {job}; cancelled sibling jobs {siblings}.", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--candidate", action="append", help="partition[,partition]/qos/4-or-8")
    submit_parser.add_argument("--time", default="08:00:00")
    submit_parser.add_argument("--replace-pending-job")
    sub.add_parser("claim")
    args = parser.parse_args()
    if args.mode == "submit":
        submit(args)
    else:
        sys.exit(claim(Path(os.environ["CAUSAL_RACE_DIR"]), os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    main()
