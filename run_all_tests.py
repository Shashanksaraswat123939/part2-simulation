import subprocess, sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent / "tests"  # tests live in tests/ subfolder

# DISCOVERED, not hand-listed. The list used to name 11 modules while tests/
# held 12: test_merge_results.py was added later and was never run by this
# runner, which still printed "0 failed" and exited 0. Its five tests cover the
# final ranking and the cross-shard merge guard -- the pipeline's output stage.
#
# Same defect as test_objective_convergence.py's hand-written __main__, which
# sat above four tests appended after it. A hand-maintained list of what to run
# silently stops covering whatever is added next; globbing cannot.
tests = sorted(p.stem for p in TEST_DIR.glob("test_*.py"))
print(f"discovered {len(tests)} test modules in {TEST_DIR}")
total_p, total_f = 0, 0
any_infra_failure = False
for t in tests:
    test_path = TEST_DIR / f'{t}.py'
    if not test_path.exists():
        print(f'{t}: ERROR - file not found at {test_path}')
        any_infra_failure = True
        continue
    r = subprocess.run([sys.executable, str(test_path)], capture_output=True, text=True)
    lines = r.stdout.strip().split('\n') if r.stdout.strip() else []
    last = lines[-1] if lines else ''
    if r.returncode != 0 and 'passed' not in last:
        print(f'{t}: INFRA FAILURE (exit {r.returncode}) - {r.stderr.strip()[:300]}')
        any_infra_failure = True
        continue
    print(f'{t}: {last}')
    if 'passed' in last and 'failed' in last:
        try:
            p = int(last.split('passed')[0].strip().split()[-1])
            f = int(last.split('failed')[0].strip().split()[-1])
            total_p += p
            total_f += f
        except (ValueError, IndexError):
            print(f'{t}: WARNING - could not parse summary line: {last!r}')
            any_infra_failure = True
print(f'\nTOTAL: {total_p} passed, {total_f} failed')
if any_infra_failure:
    print('WARNING: one or more test files could not be run at all -- counts above are incomplete.')
    sys.exit(2)
sys.exit(1 if total_f else 0)