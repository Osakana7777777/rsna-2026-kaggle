"""Exercise mixed-model inference isolation and submission alignment."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    'submission_ensemble_builder',
    Path(__file__).resolve().parents[1] / 'scripts/build_submission_ensemble.py',
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def run_fixture(tmp_path: Path, *, changed_manifest: bool = False,
                invalid_prediction: bool = False, nested_mount: bool = False) -> None:
    sample = (tmp_path / 'input' / ('competitions' if nested_mount else '')
              / 'rsna-knee-abnormality-detection/sample_submission.csv')
    sample.parent.mkdir(parents=True)
    sample.write_text('StudyInstanceUID,ACL\na,0\nb,0\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'version': 2 if changed_manifest else 1}))
    members = []
    for name, rows in [('first', 'a,0.2\nb,0.8'),
                       ('second', 'b,0.4\na,' + ('nan' if invalid_prediction else '0.6'))]:
        embedded = {'config.toml': f'[inference]\nensemble_manifest_path = "{manifest}"\n'}
        csv_text = 'StudyInstanceUID,ACL\n' + rows + '\n'
        source = (
            'from pathlib import Path\n'
            f'EMBEDDED_FILES: dict[str, str] = {embedded!r}\n'
            'CONFIG_RELATIVE_PATH = "config.toml"\n'
            'p = Path("/kaggle/working/outputs")\n'
            'p.mkdir(parents=True, exist_ok=True)\n'
            f'(p / "submission.csv").write_text({csv_text!r})\n'
        )
        source = source.replace('/kaggle/working', str(tmp_path / 'working'))
        members.append((name, source, {'version': 1}))
    runtime = builder.RUNTIME.replace('/kaggle/working', str(tmp_path / 'working'))
    runtime = runtime.replace('/tmp/rsna2026-', str(tmp_path / 'worker-'))
    runtime = runtime.replace('/kaggle/input', str(tmp_path / 'input'))
    exec('import ast\nMEMBERS = ' + repr(members) + '\n' + runtime, {})


@pytest.mark.parametrize('nested_mount', [False, True])
def test_blend_aligns_studies_and_preserves_template_order(tmp_path, nested_mount) -> None:
    run_fixture(tmp_path, nested_mount=nested_mount)
    result = pd.read_csv(tmp_path / 'working/submission.csv')
    assert result.StudyInstanceUID.tolist() == ['a', 'b']
    assert result.ACL.tolist() == pytest.approx([0.4, 0.6])
    assert (tmp_path / 'working/submission.csv').read_bytes() == (
        tmp_path / 'working/outputs/submission.csv').read_bytes()


def test_blend_stops_when_original_weights_are_not_mounted(tmp_path) -> None:
    with pytest.raises(ValueError, match='differs from original submission'):
        run_fixture(tmp_path, changed_manifest=True)


def test_blend_rejects_nonfinite_predictions(tmp_path) -> None:
    with pytest.raises(ValueError, match='Invalid member predictions'):
        run_fixture(tmp_path, invalid_prediction=True)
