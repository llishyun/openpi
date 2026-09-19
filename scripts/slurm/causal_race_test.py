import argparse
import concurrent.futures
import json
import subprocess

import pytest

from scripts.slurm import causal_race


def test_simultaneous_claims_start_exactly_one_job(tmp_path, monkeypatch):
    (tmp_path / "jobs.json").write_text(json.dumps(["101", "102"]))
    cancelled = []

    def run(argv, **kwargs):
        cancelled.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(causal_race.subprocess, "run", run)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda job: causal_race.claim(tmp_path, job), ["101", "102"]))
    assert sorted(results) == [0, 10]
    winner = json.loads((tmp_path / "winner.json").read_text())["job_id"]
    assert cancelled == [["scancel", "102" if winner == "101" else "101"]]
    assert causal_race.claim(tmp_path, winner) == 10  # no duplicate run after requeue
    with pytest.raises(ValueError, match="not a member"):
        causal_race.claim(tmp_path, "999")  # unrelated jobs cannot win/cancel


def test_held_submission_publishes_manifest_before_release(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EXP_NAME", "race")
    calls = []
    envs = []

    def run(argv, **kwargs):
        calls.append(argv)
        envs.append(kwargs["env"])
        assert "--hold" in argv
        return subprocess.CompletedProcess(argv, 0, str(100 + len(envs)) + "\n", "")

    def command(*argv):
        assert argv == ("scontrol", "release", "101", "102")
        assert json.loads((tmp_path / ".causal_races/race/jobs.json").read_text()) == ["101", "102"]
        calls.append(list(argv))
        return ""

    monkeypatch.setattr(causal_race.subprocess, "run", run)
    monkeypatch.setattr(causal_race, "command", command)
    args = argparse.Namespace(
        candidate=["A100-80GB/hpgpu/8", "A100-80GB/hpgpu/4"], time="08:00:00", replace_pending_job=None
    )
    causal_race.submit(args)
    assert [e["GPU_COUNT"] for e in envs] == ["8", "4"]
    assert all(e["SMOKE_TEST"] == "0" and e["RESUME"] == "0" for e in envs)


def test_old_job_starts_during_submission_new_jobs_cancelled(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EXP_NAME", "race")
    states = iter(["PENDING", "RUNNING"])
    monkeypatch.setattr(causal_race, "pending_state", lambda job: next(states))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "101\n", "")

    monkeypatch.setattr(causal_race.subprocess, "run", run)
    args = argparse.Namespace(candidate=["A100-80GB/hpgpu/4"], time="08:00:00", replace_pending_job="969068")
    with pytest.raises(RuntimeError, match="Old job started"):
        causal_race.submit(args)
    assert calls[-1] == ["scancel", "101"]
    assert not any("969068" in argv for argv in calls)


def test_rejected_candidates_leave_old_job_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EXP_NAME", "race")
    monkeypatch.setattr(causal_race, "pending_state", lambda job: "PENDING")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, "", "Invalid qos")

    monkeypatch.setattr(causal_race.subprocess, "run", run)
    args = argparse.Namespace(candidate=["A100-80GB/hpgpu/4"], time="08:00:00", replace_pending_job="969068")
    with pytest.raises(RuntimeError, match="No candidate accepted"):
        causal_race.submit(args)
    assert all(argv[0] == "sbatch" for argv in calls)
