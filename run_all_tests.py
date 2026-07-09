import subprocess, sys
tests = [
    'test_physics_contract', 'test_mass_com_ingest', 'test_cfd_wrapper',
    'test_mesh_validation', 'test_calibration', 'test_candidate_record',
    'test_race_objective_adapter', 'test_adjoint_contract', 'test_integration_end_to_end'
]
total_p, total_f = 0, 0
for t in tests:
    r = subprocess.run([sys.executable, f'tests/{t}.py'], capture_output=True, text=True)
    lines = r.stdout.strip().split('\n')
    last = lines[-1] if lines else f'ERROR: {r.stderr[:200]}'
    print(f'{t}: {last}')
    if 'passed' in last and 'failed' in last:
        p = int(last.split('passed')[0].strip().split()[-1])
        f = int(last.split('failed')[0].strip().split()[-1])
        total_p += p
        total_f += f
print(f'\nTOTAL: {total_p} passed, {total_f} failed')