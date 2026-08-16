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
    assert "BEST" in r.stdout and "d_halo=30.00" in r.stdout, (
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


def test_record_carries_drag_and_mass_for_the_ranked_table():
    """A ranking showing only T_raw cannot be judged.

    The first real record produced empty D20 and mass columns, and those are
    exactly what tells you WHY one car beat another -- lighter is a different
    story from slipperier. The inner loop holds both at write time; the record
    just never asked. Duck-typed shapes, matching what Part 3 actually has.
    """
    from types import SimpleNamespace

    cfd = SimpleNamespace(D20=0.7076, L=-0.02, Cm=0.005, A=0.0075)
    mass = SimpleNamespace(total_mass_kg=0.1586, com_x_m=0.10,
                           com_y_m=0.0, com_z_m=0.0292)   # no .components
    rec = CandidateRecord(
        candidate_id="c1", W_mm=120.0, x_front_mm=42.9, d_halo_mm=16.0,
        lifecycle_state="geometry_repaired", T_raw=3.0095, T_penalized=3.0096,
        cfd_force_report=cfd, mass_report=mass, com_report=mass)
    with tempfile.TemporaryDirectory() as td:
        p = write_candidate_record(rec, td)
        back = read_candidate_record(p)
        assert abs(back.cfd_force_report.D20 - 0.7076) < 1e-12
        assert abs(back.mass_report.total_mass_kg - 0.1586) < 1e-12
        r = subprocess.run([sys.executable, str(_ROOT / "merge_results.py"), td],
                           capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0.70760" in r.stdout, f"D20 missing from the table:\n{r.stdout}"
    assert "158.60" in r.stdout, f"mass missing from the table:\n{r.stdout}"


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


def test_ranking_uses_T_penalized_and_drops_underweight_cars():
    """The last step must answer the question the optimiser was solving.

    merge_results sorted on T_raw, discarding every penalty the pipeline had
    just computed -- COM, manufacturing, and the T3.6 minimum mass. On the
    2026-08-11 sweep that named a 45.97 g car the winner: two grams under the
    48 g floor, illegal, and beaten on T_penalized by a legal car it was only
    "faster" than on raw time.

    The mass floor is the harder of the two tests. Its penalty is sized to
    steer the DESCENT; it is not a price a candidate may pay to win, so an
    underweight car is excluded from the ranking rather than merely charged.
    """
    import json
    import subprocess
    import sys
    import tempfile

    from candidate_record import CandidateRecord, write_candidate_record
    from physics_contract import FullCarMassCOM

    def rec(cid, d, t_raw, t_pen, grams):
        return CandidateRecord(
            candidate_id=cid, W_mm=120.0, x_front_mm=42.9, d_halo_mm=d,
            lifecycle_state="geometry_repaired", T_raw=t_raw, T_penalized=t_pen,
            mass_report=FullCarMassCOM(total_mass_kg=grams / 1000.0,
                                       com_x_m=0.12, com_y_m=0.0, com_z_m=0.03))

    with tempfile.TemporaryDirectory() as td:
        # Fastest on RAW time, but 2 g underweight -> must not win, must not rank.
        write_candidate_record(rec("illegal", 44.0, 1.4754, 1.6834, 68.97), td)
        # Slower raw, better penalised, and legal.
        write_candidate_record(rec("legal_a", 16.0, 1.4811, 1.5095, 71.14), td)
        write_candidate_record(rec("legal_b", 30.0, 1.4790, 1.5311, 71.02), td)
        r = subprocess.run([sys.executable, str(_ROOT / "merge_results.py"), td],
                           capture_output=True, text=True, timeout=300)

    assert r.returncode == 0, f"merge failed: {r.stdout}\n{r.stderr}"
    assert "d_halo=16.00" in r.stdout, (
        f"should win on T_penalized (1.5095 < 1.5311):\n{r.stdout}")
    assert "44.00" not in r.stdout.split("BEST")[-1], (
        "the underweight car must not be the answer")
    assert "EXCLUDED" in r.stdout and "underweight" in r.stdout, (
        f"exclusion must be reported, not silent:\n{r.stdout}")
    assert "45.97" in r.stdout, "the excluded car's mass should be shown"


if __name__ == "__main__":
    # Collected BY NAME. The hand-written call list this replaces skipped every
    # test appended below it -- which is exactly what happened to
    # test_ranking_uses_T_penalized_and_drops_underweight_cars, and what
    # test_no_test_file_silently_skips_its_own_tests exists to catch.
    _mod = sys.modules[__name__]
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        _run(getattr(_mod, _n))
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
