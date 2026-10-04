"""Package the exact submission 0019 and 0021 inference sources for blending."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

RUNTIME = '''from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd

OUTPUT = Path('/kaggle/working/outputs')
OUTPUT.mkdir(parents=True, exist_ok=True)
records = []
for name, source, expected in MEMBERS:
    tree = ast.parse(source)
    embedded = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == 'EMBEDDED_FILES')
    config_key = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CONFIG_RELATIVE_PATH'
                              for t in node.targets))
    import tomllib
    config = tomllib.loads(embedded[config_key])
    manifest_path = Path(config['inference']['ensemble_manifest_path'])
    actual = json.loads(manifest_path.read_text())
    if actual != expected:
        raise ValueError(f'{name}: mounted ensemble differs from original submission')
    group_output = OUTPUT / name
    source = source.replace('/kaggle/working/outputs', str(group_output))
    source = source.replace('/tmp/rsna2026-payload', f'/tmp/rsna2026-{name}-payload')
    script = Path(f'/tmp/rsna2026-{name}-inference.py')
    script.write_text(source)
    subprocess.run([sys.executable, str(script)], check=True)
    table = pd.read_csv(group_output / 'submission.csv', dtype={'StudyInstanceUID': str})
    if not table['StudyInstanceUID'].is_unique:
        raise ValueError('Duplicate study identifiers')
    values = table.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid member predictions')
    records.append(table)
first, second = records
if list(first.columns) != list(second.columns) or set(first.StudyInstanceUID) != set(second.StudyInstanceUID):
    raise ValueError('Member submission schemas or studies differ')
second = second.set_index('StudyInstanceUID').loc[first.StudyInstanceUID].reset_index()
result = first.copy()
result.iloc[:, 1:] = (first.iloc[:, 1:].to_numpy() + second.iloc[:, 1:].to_numpy()) / 2
sample_candidates = [Path('/kaggle/input') / prefix / 'rsna-knee-abnormality-detection' / 'sample_submission.csv'
                     for prefix in ['', 'competitions']]
sample_path = next((p for p in sample_candidates if p.is_file()), None)
if sample_path is None:
    raise FileNotFoundError('Competition sample submission is not mounted')
sample = pd.read_csv(sample_path, dtype={'StudyInstanceUID': str})
if list(result.columns) != list(sample.columns) or set(result.StudyInstanceUID) != set(sample.StudyInstanceUID):
    raise ValueError('Ensemble differs from competition submission template')
result = result.set_index('StudyInstanceUID').loc[sample.StudyInstanceUID].reset_index()
result.to_csv(OUTPUT / 'submission.csv', index=False)
Path('/kaggle/working/submission.csv').write_bytes((OUTPUT / 'submission.csv').read_bytes())
summary = {'status': 'inference-complete', 'method': 'equal-mean-of-four-fold-rank-means',
           'weights': [0.5, 0.5], 'test_studies': len(result),
           'submission_sha256': hashlib.sha256((OUTPUT / 'submission.csv').read_bytes()).hexdigest(),
           'members': [{'name': n, 'source_sha256': hashlib.sha256(s.encode()).hexdigest(),
                        'ensemble': e} for n, s, e in MEMBERS]}
(OUTPUT / 'run-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
'''


def build(source_dir: Path, output_dir: Path) -> None:
    members = []
    for name, version, training in [
        ('dinov2-0019', 31, 'rsna-2026-training-v45'),
        ('resnet-0021', 33, 'rsna-2026-radimagenet-resnet50-v1'),
    ]:
        notebook = json.loads((source_dir / f'source-v{version}.ipynb').read_text())
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        if len(code_cells) != 1:
            raise ValueError('Expected one original launcher cell')
        source = code_cells[0]['source']
        source = ''.join(source) if isinstance(source, list) else source
        if version == 31:
            source = source.replace(
                '/kaggle/input/rsna-2026-training/outputs/ensemble-manifest.json',
                '/kaggle/input/rsna-2026-dinov2-ranking-v45/ensemble-manifest.json',
            )
        compile(source, f'v{version}', 'exec')
        manifest = json.loads((source_dir / training / 'outputs/ensemble-manifest.json').read_text())
        members.append((name, source, manifest))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 'import ast\nMEMBERS = ' + repr(members) + '\n' + RUNTIME
    compile(generated, 'ensemble.py', 'exec')
    (output_dir / 'ensemble.py').write_text(generated)
    metadata = json.loads(Path('kaggle/ensemble-0019-0021-kernel-metadata.json').read_text())
    (output_dir / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('outputs/ensemble-0019-0021'))
    parser.add_argument('--output-dir', type=Path, default=Path('.kaggle-build/ensemble-0019-0021'))
    args = parser.parse_args()
    build(args.source_dir, args.output_dir)
