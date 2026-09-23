#!/usr/bin/env bash
# Build a data-free, verified current-upstream submission directory and ZIP.
# Usage: scripts/make_submission_pack.sh OUT [--results DIR] [--allow-smoke]
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$R/evaluation/submission-20260919/upstream/.venv/bin/python}"
exec "$PY" - "$R" "$@" <<'PY'
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

root = Path(sys.argv.pop(1)).resolve()
relative = Path('evaluation/submission-20260919')
here = root / relative
parser = argparse.ArgumentParser(description='Package a complete, data-free nested evaluation; never reuse historical fitted models.')
parser.add_argument('output', type=Path)
parser.add_argument('--results', type=Path, default=here / 'run')
parser.add_argument('--allow-smoke', action='store_true', help='QA only: requires an output name ending in -smoke; visibly marks partial coverage')
parser.add_argument('--allow-deferred', action='store_true', help='archive a run that explicitly defers frozen users; adds a visible coverage-gap file')
args = parser.parse_args()
output = args.output.expanduser().absolute()
archive = Path(str(output) + '.zip')
sidecar = Path(str(archive) + '.sha256')
for path in (output, archive, sidecar):
    if path.exists() or path.is_symlink():
        parser.error(f'refusing to overwrite {path}')
output = output.resolve()
if output.is_relative_to(root) or root.is_relative_to(output):
    parser.error('output must be outside the source checkout')
results = args.results.resolve()
manifest = json.loads((results / 'manifest.json').read_text())
protocol = json.loads((here / 'protocol.json').read_text())
def user_ids(path):
    payload = json.loads(Path(path).read_text())
    return sorted(int(u) for u in (payload['user_ids'] if isinstance(payload, dict) else payload))

expected = user_ids(here / 'users.json')
users = manifest['users']
deferred = manifest.get('deferred_users') or []
if deferred and not args.allow_deferred:
    parser.error(
        f'{len(deferred)} frozen users are deferred ({deferred}); pass --allow-deferred to archive a '
        'release with a visible coverage gap'
    )
if not args.allow_smoke and (
    users != [u for u in expected if u not in set(deferred)]
    or manifest['rounds'] != protocol['booster']['num_boost_round']
):
    parser.error('incomplete evaluation or non-frozen training rounds; refusing a release')
if args.allow_smoke and not output.name.endswith('-smoke'):
    parser.error('QA artifacts must have an output name ending in -smoke')
if len(users) < 12 or users != sorted(set(users)) or not set(users) <= set(expected):
    parser.error('invalid evaluation user coverage')
report = json.loads((results / 'results.json').read_text())
if report['users'] != len(users):
    parser.error('result count differs from the run manifest')
if not args.allow_smoke and not deferred and report['reviews'] != protocol['expected_scored_reviews']:
    parser.error('incomplete scored-review coverage')
if deferred and report.get('deferred_users') != deferred:
    parser.error('run report does not record the same deferred users as the manifest')

files = {}
def add(source, destination=None):
    source = Path(source)
    destination = Path(destination) if destination is not None else source.relative_to(root)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f'missing or symlinked input: {source}')
    if destination.is_absolute() or '..' in destination.parts:
        raise ValueError(f'unsafe archive path: {destination}')
    files[destination] = source

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

for name in ('baseline.py', 'nested.py', 'run.py', 'verify_submission.py', 'test_baseline.py',
             'test_submission.py', 'protocol.json', 'users.json', 'upstream-reference.jsonl',
             'LICENSE', 'THIRD-PARTY-NOTICES.md', 'README.md', 'AUDIT.md',
             'baseline-verification.json', 'smoke-verification.json', 'pyrefly.toml'):
    add(here / name)
for name in ('README.md', 'AUDIT.md', 'LICENSE', 'THIRD-PARTY-NOTICES.md'):
    add(here / name, name)
for name in ('probe.py', 'serve.py', 'feature-spec.json', 'feature-spec-fresh.json'):
    add(root / 'evaluation/tabular-probe-20260917' / name)
add(root / 'evaluation/probe-stage2-20260918/stage2.py')
add(root / 'scripts/make_submission_pack.sh')
for name in ('manifest.json', 'splits.json', 'results.json', 'result-FSRS-7.jsonl',
             'result-B.jsonl', 'result-FSRS7-ResidualBoost.jsonl'):
    add(results / name, relative / 'run' / name)
for fold in range(protocol['outer_folds']):
    for name in ('model.json', 'fit.json', 'calibration.json', 'lineage.json'):
        add(results / f'fold-{fold}' / name, relative / 'run' / f'fold-{fold}' / name)
for user in users:
    for kind in ('base', 'scores'):
        path = results / kind / f'{user}.json'
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'missing or symlinked provenance: {path}')
for name, expected_hash in manifest['source_sha256'].items():
    if digest(root / name) != expected_hash:
        raise ValueError(f'training source changed: {name}')
requirements = '\n'.join(f'{name}=={version(name)}' for name in
                         ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'xgboost', 'pytest')) + '\n'

# All checks above precede output creation. A failed verification leaves no published archive.
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
    stage = Path(temporary) / output.name
    for destination, source in files.items():
        target = stage / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    destination = stage / relative / 'run'
    public_manifest = json.loads((destination / 'manifest.json').read_text())
    public_manifest['original_manifest_sha256'] = digest(results / 'manifest.json')
    public_manifest['data'] = '<dataset root is not distributed; pass --data>'
    (destination / 'manifest.json').write_text(json.dumps(public_manifest, indent=2) + '\n')
    for fold in range(protocol['outer_folds']):
        fit_path = destination / f'fold-{fold}/fit.json'
        fit = json.loads(fit_path.read_text())
        fit['original_fit_sha256'] = digest(results / f'fold-{fold}/fit.json')
        fit['model'] = 'model.json'
        fit['calibration_json'] = 'calibration.json'
        fit['lineage_json'] = 'lineage.json'
        fit_path.write_text(json.dumps(fit, indent=2, allow_nan=False) + '\n')
    for name in ('feature-spec.json', 'feature-spec-fresh.json'):
        spec_path = stage / 'evaluation/tabular-probe-20260917' / name
        spec = json.loads(spec_path.read_text())
        spec_path.write_text(json.dumps({'features_flat': spec['features_flat']}, indent=2) + '\n')
    (stage / 'requirements.txt').write_text(requirements)
    if deferred:
        gap = (
            f'COVERAGE GAP: {len(deferred)} of {manifest["frozen_users"]} frozen users deferred.\n'
            f'Deferred user IDs: {deferred}\nReason: {manifest.get("defer_reason")}\n'
            f'Reported coverage: {report["users"]} users / {report["reviews"]} scored reviews.\n'
            'The frozen protocol is unchanged. A complete result requires a NEW output directory\n'
            'with all baseline caches, a full two-fold refit, score, report and re-archive.\n'
            'Adding deferred users reshuffles the folds; a report-only rerun is invalid.\n'
        )
        (stage / 'COVERAGE-GAP.txt').write_text(gap)
        (stage / 'README.md').write_text(gap + '\n' + (stage / 'README.md').read_text())
    if args.allow_smoke:
        warning = f'SMOKE ONLY: {len(users)} users, {manifest["rounds"]} rounds. Not a full benchmark or a submission.\n'
        (stage / 'SMOKE-ONLY.txt').write_text(warning)
        (stage / 'README.md').write_text(warning + '\n' + (stage / 'README.md').read_text())
    destination = stage / relative / 'run'
    with (destination / 'base-provenance.jsonl').open('w') as base_file, (destination / 'score-provenance.jsonl').open('w') as score_file:
        for user in users:
            base = json.loads((results / 'base' / f'{user}.json').read_text())
            public = {key: base[key] for key in ('user', 'rows', 'status', 'identity', 'identity_fields', 'cache_sha256', 'elapsed_seconds')}
            public['metrics'] = base['stats']['metrics']
            base_file.write(json.dumps(public, allow_nan=False) + '\n')
            score = json.loads((results / 'scores' / f'{user}.json').read_text())
            if set(score) != {'identity', 'records'} or any(set(record) != {'user', 'size', 'metrics'} for record in score['records'].values()):
                raise ValueError(f'unexpected fields in score provenance: {user}')
            score_file.write(json.dumps(score, allow_nan=False) + '\n')
    table = '| Model | LogLoss | RMSE(bins) | AUC |\n|---|---:|---:|---:|\n'
    for model, metrics in report['models'].items():
        table += '| ' + model + ' | ' + ' | '.join(f'{metrics[key]["mean"]:.6f}' for key in ('LogLoss', 'RMSE(bins)', 'AUC')) + ' |\n'
    draft = ('# Proposed FSRS7-ResidualBoost benchmark evaluation\n\n'
             + ('**SMOKE ONLY — NOT FOR SUBMISSION.**\n\n' if args.allow_smoke else '')
             + f'Coverage: {report["users"]:,} users / {report["reviews"]:,} scored reviews. Current upstream pin: `{manifest["upstream_revision"]}`.\n\n'
             + table
             + '\nTwo outer user folds; inner-cross-fitted three-coefficient calibration; equal-user boosted correction. Validation and held-user labels never fit the calibration used for those users. Fresh chronological upstream FSRS-7 predictions; no historical fitted corrections or boosters reused.\n\n'
             + 'This fixes cross-stage fitting leakage, not prior model-selection exposure: the public dataset was studied previously. These are instantaneous recall predictions, not evidence of improved scheduling. This cross-user training protocol is supplied as a standalone runner, not represented as an already-merged upstream algorithm registration.\n\n'
             + 'The archive contains MIT-authorized project source and trained artifacts, per-user aggregate results and provenance, SHA256 checksums, a data-free verifier, and complete reproduction commands. Dataset and upstream rights remain separate. Raw reviews and per-review caches are deliberately absent.\n')
    leaked = [
        str(path.relative_to(stage))
        for path in stage.rglob('*')
        if path.is_file() and str(root).encode() in path.read_bytes()
    ]
    if leaked:
        raise ValueError(f'machine-specific checkout path leaked into release: {leaked}')
    (stage / 'ISSUE-DRAFT.md').write_text(draft)
    inventory = sorted(path for path in stage.rglob('*') if path.is_file())
    (stage / 'SHA256SUMS').write_text(''.join(f'{digest(path)}  {path.relative_to(stage)}\n' for path in inventory))
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    subprocess.run([sys.executable, str(stage / relative / 'verify_submission.py'), '--root', str(stage)], cwd=stage, env=environment, check=True)
    temporary_zip = Path(temporary) / 'release.zip'
    with zipfile.ZipFile(temporary_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                package.write(path, Path(output.name) / path.relative_to(stage))
    with tempfile.TemporaryDirectory(prefix='extracted-', dir=temporary) as extraction:
        with zipfile.ZipFile(temporary_zip) as package:
            if package.testzip() is not None:
                raise ValueError('ZIP integrity check failed')
            package.extractall(extraction)
        extracted = Path(extraction) / output.name
        subprocess.run([sys.executable, str(extracted / relative / 'verify_submission.py'), '--root', str(extracted)], cwd=extracted, env=environment, check=True)
    archive_hash = digest(temporary_zip)
    stage.rename(output)
    temporary_zip.rename(archive)
    sidecar.write_text(f'{archive_hash}  {archive.name}\n')
print(f'{"SMOKE ONLY" if args.allow_smoke else "Submission"}: {archive}')
print(f'SHA256: {archive_hash}')
PY
