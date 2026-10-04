"""Retrieve pinned source and validation artifacts for submissions 0019/0021.

Run with the Python interpreter from a uv-managed Kaggle CLI environment.
The SDK version_label must be v<number>; current CLI version suffix handling
cannot reliably retrieve these historical artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelRequest,
    ApiListKernelSessionOutputRequest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-dinov2', action='store_true')
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    base = Path('outputs/ensemble-0019-0021')
    base.mkdir(parents=True, exist_ok=True)
    with api.build_kaggle_client() as client:
        for version in (31, 33):
            request = ApiGetKernelRequest()
            request.user_name = 'osakanateishoku'
            request.kernel_slug = 'rsna-2026-workflow'
            request.version_label = f'v{version}'
            response = client.kernels.kernels_api_client.get_kernel(request)
            (base / f'source-v{version}.ipynb').write_text(response.blob.source)
        for slug, version in [
            ('rsna-2026-training', 45),
            ('rsna-2026-radimagenet-resnet50', 1),
        ]:
            token = None
            while True:
                request = ApiListKernelSessionOutputRequest()
                request.user_name = 'osakanateishoku'
                request.kernel_slug = slug
                request.version_label = f'v{version}'
                request.page_size = 200
                request.page_token = token
                response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                for item in response.files or []:
                    selected = item.file_name in {
                        'outputs/oof-predictions.csv',
                        'outputs/ensemble-manifest.json',
                        'outputs/cv-summary.json',
                    }
                    checkpoint = (args.freeze_dinov2 and version == 45
                                  and item.file_name.endswith(('best-model.pt', 'checkpoint-manifest.json')))
                    if not selected and not checkpoint:
                        continue
                    path = base / f'{slug}-v{version}' / item.file_name
                    if checkpoint:
                        path = base / 'dinov2-v45-dataset' / item.file_name.removeprefix('outputs/')
                    download = requests.get(item.url, timeout=120)
                    download.raise_for_status()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(download.content)
                    print(path)
                token = response.next_page_token
                if not token:
                    break
    if args.freeze_dinov2:
        root = base / 'dinov2-v45-dataset'
        manifest = json.loads((base / 'rsna-2026-training-v45/outputs/ensemble-manifest.json').read_text())
        for member in manifest['members']:
            actual = hashlib.sha256((root / member['checkpoint_file']).read_bytes()).hexdigest()
            if actual != member['checkpoint_sha256']:
                raise ValueError('Original DINOv2 checkpoint hash mismatch')
        (root / 'ensemble-manifest.json').write_text(json.dumps(manifest, indent=2))
        (root / 'dataset-metadata.json').write_text(json.dumps({
            'id': 'osakanateishoku/rsna-2026-dinov2-ranking-v45',
            'title': 'RSNA 2026 DINOv2 ranking v45 frozen checkpoints',
            'licenses': [{'name': 'other'}],
        }, indent=2))


if __name__ == '__main__':
    main()
