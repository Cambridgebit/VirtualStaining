import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

from evaluations.inception import InceptionV3
from prdc import compute_prdc

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def imread(filename):
    """Load image to (H, W, 3) uint8 array, dropping alpha if present."""
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def get_activations(files, model, batch_size=64, dims=2048, device=None):
    """Compute Inception activations for given image file paths."""
    model.eval()

    if len(files) == 0:
        raise ValueError('No images found to compute activations.')
    if batch_size > len(files):
        batch_size = len(files)

    pred_arr = np.empty((len(files), dims), dtype=np.float32)

    for i in tqdm(range(0, len(files), batch_size)):
        start = i
        end = min(i + batch_size, len(files))

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])
        images = images.transpose((0, 3, 1, 2))
        images /= 255.0

        batch = torch.from_numpy(images).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().numpy().reshape(pred.size(0), -1)

    return pred_arr


def _list_image_files(path: pathlib.Path):
    files = []
    for ext in IMG_EXTENSIONS:
        files += list(path.glob(f'*{ext}'))
    return files


def _compute_activations_of_path(path, model, batch_size, dims, device):
    path = pathlib.Path(path)
    files = _list_image_files(path)
    if len(files) == 0:
        raise RuntimeError(f'No images found in: {path}')
    f = get_activations(files, model, batch_size, dims, device)
    return f


def calculate_prdc_given_paths(paths, batch_size=64, device=None, dims=2048, k=95):
    """Compute PRDC metrics between two image directories."""
    for p in paths:
        if not os.path.isdir(p):
            raise RuntimeError(f'Invalid directory: {p}')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    f_real = _compute_activations_of_path(paths[0], model, batch_size, dims, device)
    f_fake = _compute_activations_of_path(paths[1], model, batch_size, dims, device)

    metrics = compute_prdc(real_features=f_real, fake_features=f_fake, nearest_k=k)
    return metrics


def _resolve_device(device_str: str | None):
    if device_str in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def main():
    parser = ArgumentParser(description='Compute PRDC metrics between two folders')
    parser.add_argument('paths', type=str, nargs=2, help='Two directories of images: real, fake')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for Inception forward pass')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help='Inception feature dimensionality (default: 2048)')
    parser.add_argument('--device', type=str, default='auto', help="Device: 'auto', 'cpu', 'cuda', or 'cuda:0'")
    parser.add_argument('--k', type=int, default=95, help='Neighbor count for PRDC')

    args = parser.parse_args()
    device = _resolve_device(args.device)

    metrics = calculate_prdc_given_paths(args.paths,
                                         batch_size=args.batch_size,
                                         device=device,
                                         dims=args.dims,
                                         k=args.k)
    # Expected keys: precision, recall, density, coverage
    for key, val in metrics.items():
        print(f'{key}: {val:.6f}')


if __name__ == '__main__':
    main()
