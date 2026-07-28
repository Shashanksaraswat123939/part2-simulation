"""
test_merge_results.py -- the record round trip and the shard-merge guard.

Two things had never been exercised on a real record:
  * write -> read. write_record raised TypeError on every call until
    2026-07-28, so no record existed to read back, and read_candidate_record
    was called only from tests using hand-built fixtures.
  * merging across shards. Nothing consumed records at all, so a sweep's
    output had no reader and no ranking.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_P2 = _HERE.parent
_ROOT = _P2.parent
sys.path.insert(0, str(_P2))

from candidate_record import (CandidateRecord, read_candidate_record,  # noqa: E402
                             write_candidate_record)

_passed = _failed = 0


def _run(t):
    global _passed, _failed
    try:
        t()
        print(f"PASS {t.__name__}")
        _passed += 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {t.__name__}: {exc!r}")
        _failed += 1


def _summary_record(cid, d_halo, t_raw, W=120.0, x_front=42.9):
    """Exactly what the inner loop emits -- no physics blocks."""
    return CandidateRecord(
        candidate_id=cid, W_mm=W, x_front_mm=x_front, d_halo_mm=d_halo,
        lifecycle_state="geometry_repaired", T_raw=t_raw,
        T_penalized=t_raw + 0.00005,
    )


def test_summary_record_round_trips():
    """The reader must read what the writer emits.

    The physics-heavy blocks are None on a summary record; read_candidate_record
    indexed them directly and would have raised KeyError/TypeError on the first
    real record ever written.
    """
    with tempfile.TemporaryDirectory() as td:
        p = write_candidate_record(_summary_record("c1", 16.0, 3.21), td)
        back = read_candidate_record(p)
    assert back.candidate_id == "c1"
    assert back.x_front_mm == 42.9, "x_front_mm did not survive the round trip"
    assert back.d_halo_mm == 16.0
    assert abs(back.T_raw - 3.21) < 1e-12
    assert back.mass_report is None and back.cfd_force_report is None


def test_merge_ranks_by_race_time():
    with tempfile.TemporaryDirectory() as td:
        for cid, d, t in (("a", 16.0, 3.25), ("b", 30.0, 3.19), ("c", 44.0, 3.31)):
            write_candidate_record(_summary_record(cid, d, t), td)
        r = subprocess.run([sys.executable, str(_ROOT / "merge_results.py"), td],
                           capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"merge failed: {r.stdout}\n{r.stderr}"
    assert "BEST: d_halo=30.00" in r.stdout, (
        f"ranked the wrong candidate best:\n{r.stdout}")
    # The noise caveat must travel with the answer, not live only in a doc.
    assert "15 ms" in r.stdout, "the drag-noise caveat is missing from the output"


def test_merge_refuses_records_from_different_cars():
    """The shard bug, made impossible to ignore.

    Until --stage1-in existed, every shard re-ran Stage 1 and picked its own
    (W, x_front), so merging their records compared different cars while
    looking perfectly reasonable.
    """
    with tempfile.TemporaryDirectory() as td:
        write_candidate_record(_summary_record("a", 16.0, 3.25, W=120.0), td)
        write_candidate_record(_summary_record("b", 30.0, 3.19, W=132.0), td)
        r = subprocess.run([sys.executable, str(_ROOT / "merge_results.py"), td],
                           capture_output=True, text=True, timeout=300)
    assert r.returncode != 0, "merged records describing different cars"
    out = r.stdout + r.stderr
    assert "DIFFERENT CARS" in out, f"unclear refusal:\n{out}"
    assert "--stage1-in" in out, "refusal should name the fix"


def test_merge_reports_unscored_candidates_rather_than_dropping_them():
    with tempfile.TemporaryDirectory() as td:
        write_candidate_record(_summary_record("ok", 16.0, 3.25), td)
        dead = CandidateRecord(
            candidate_id="dead", W_mm=120.0, x_front_mm=42.9, d_halo_mm=30.0,
            lifecycle_state="CFD_failed", T_raw=None, T_penalized=None,
            failure_reason="mesh blew up")
        write_candidate_record(dead, td)
        r = subprocess.run([sys.executable, str(_ROOT / "merge_results.py"), td],
                           capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CFD_failed" in r.stdout, (
        "failed candidates must be reported, not silently dropped from the count")


if __name__ == "__main__":
    for t in (test_summary_record_round_trips,
              test_merge_ranks_by_race_time,
              test_merge_refuses_records_from_different_cars,
              test_merge_reports_unscored_candidates_rather_than_dropping_them):
        _run(t)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
