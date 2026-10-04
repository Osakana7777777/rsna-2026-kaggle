"""Evaluate the fixed 50:50 ensemble on matching held-out gold studies."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rsna2026.checkpoint import file_sha256
from rsna2026.constants import TARGETS
from rsna2026.ensemble import percentile_rank_mean
from rsna2026.train import macro_auc, target_auc_scores


def main() -> None:
    base = Path('outputs/ensemble-0019-0021')
    paths = [base / name / 'outputs/oof-predictions.csv' for name in [
        'rsna-2026-training-v45', 'rsna-2026-radimagenet-resnet50-v1',
    ]]
    tables = [pd.read_csv(p, dtype={'StudyInstanceUID': str})
              .sort_values('StudyInstanceUID').reset_index(drop=True) for p in paths]
    a, b = tables
    if not all(t.StudyInstanceUID.is_unique for t in tables):
        raise ValueError('Duplicate validation studies')
    if not a.StudyInstanceUID.equals(b.StudyInstanceUID) or not a.outer_fold.equals(b.outer_fold):
        raise ValueError('Validation studies or gold folds differ')
    labels = [t[[f'label_{x}' for x in TARGETS]].to_numpy() for t in tables]
    predictions = [t[[f'prediction_{x}' for x in TARGETS]].to_numpy() for t in tables]
    if not np.array_equal(*labels) or not all(np.isfinite(x).all() for x in labels + predictions):
        raise ValueError('Validation labels differ or contain nonfinite values')
    ranked = percentile_rank_mean(predictions)
    arrays = [*predictions, np.mean(predictions, axis=0), ranked]
    names = ['dinov2', 'resnet', 'probability_mean', 'rank_mean']
    result = {
        'studies': len(a), 'weights': [0.5, 0.5], 'matching_gold_folds': True,
        'sources': [{'path': str(p), 'sha256': file_sha256(p)} for p in paths],
        'scores': {n: macro_auc(labels[0], p) for n, p in zip(names, arrays)},
        'rank_mean_by_target': target_auc_scores(labels[0], ranked),
        'limitation': 'OOF ranks use 58 held-out studies; test ranks are averaged within each family across four models before the 50:50 blend.',
    }
    rng = np.random.default_rng(42)
    samples, deltas = [], []
    for _ in range(5000):
        ids = rng.integers(0, len(a), len(a))
        score = macro_auc(labels[0][ids], ranked[ids])
        delta = score - macro_auc(labels[0][ids], predictions[0][ids])
        if np.isfinite(score) and np.isfinite(delta):
            samples.append(score)
            deltas.append(delta)
    result['bootstrap'] = {
        'unit': 'study', 'seed': 42, 'resamples': len(samples),
        'auc_95': np.quantile(samples, [0.025, 0.975]).tolist(),
        'delta_vs_dinov2_95': np.quantile(deltas, [0.025, 0.975]).tolist(),
    }
    (base / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
