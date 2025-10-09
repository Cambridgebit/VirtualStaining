"""FID/KID evaluation utilities (PyTorch)

This module computes FID (Frechet Inception Distance) and KID (Kernel
Inception Distance) between two image sets. It uses the InceptionV3
backbone compatible with the official FID implementation to extract
features, then computes:
  - FID from Gaussian statistics of features
  - KID via an unbiased MMD^2 estimate with polynomial kernel

CLI is intentionally minimal — just pass two paths (directories of images
or .npz stats for FID). Device and other options have sensible defaults.

Original FID code adapted from https://github.com/bioinf-jku/TTUR (PyTorch port).
"""
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

from evaluations.inception import InceptionV3

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def imread(filename):
    """Load an image path into a (H, W, 3) uint8 array.

    Always returns 3 channels by dropping alpha if present.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def get_activations(files, model, batch_size=64, dims=2048, device=None):
    """Compute Inception activations for given image file paths.

    Returns an array of shape (N, dims).
    """
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
        images = images.transpose((0, 3, 1, 2))  # (N,3,H,W)
        images /= 255.0

        batch = torch.from_numpy(images).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial avg pooling
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().numpy().reshape(pred.size(0), -1)

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=64, dims=2048,
                                    device=None):
    """Compute mean and covariance of Inception activations for FID."""
    act = get_activations(files, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _list_image_files(path: pathlib.Path):
    files = []
    for ext in IMG_EXTENSIONS:
        files += list(path.glob(f'*{ext}'))
    return files


def _compute_statistics_of_path(path, model, batch_size, dims, device):
    """Return (mu, sigma) for a directory of images or an .npz stats file."""
    if str(path).lower().endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = _list_image_files(path)
        if len(files) == 0:
            raise RuntimeError(f'No images found in: {path}')
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device)

    return m, s


def _compute_activations_of_path(path, model, batch_size, dims, device):
    """Return activations array for a directory of images.

    KID requires raw activations. If path is an .npz, this is unsupported.
    """
    if str(path).lower().endswith('.npz'):
        raise ValueError('KID requires image directories, not .npz stats files.')
    path = pathlib.Path(path)
    files = _list_image_files(path)
    if len(files) == 0:
        raise RuntimeError(f'No images found in: {path}')
    return get_activations(files, model, batch_size, dims, device)


def calculate_fid_given_paths(paths, batch_size=64, device=None, dims=2048):
    """Compute FID between two paths (dirs of images or .npz stats)."""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError(f'Invalid path: {p}')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, device)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return float(fid_value)


def _poly_kernel(x, y=None, degree=3, gamma=None, coef0=1.0):
    """Polynomial kernel <gamma * x·y + coef0>^degree.

    If y is None, uses x vs x. gamma defaults to 1/d.
    """
    x = np.atleast_2d(x)
    y = x if y is None else np.atleast_2d(y)
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * np.dot(x, y.T) + coef0) ** degree


def _mmd2_unbiased(k_xx, k_yy, k_xy):
    """Unbiased MMD^2 estimate using precomputed kernels."""
    m = k_xx.shape[0]
    n = k_yy.shape[0]
    # Exclude diagonal terms
    sum_xx = (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1))
    sum_yy = (np.sum(k_yy) - np.trace(k_yy)) / (n * (n - 1))
    sum_xy = np.sum(k_xy) / (m * n)
    return sum_xx + sum_yy - 2.0 * sum_xy


def calculate_kid_given_paths(paths, batch_size=64, device=None, dims=2048,
                              subset_size=1000, subsets=50,
                              degree=3, gamma=None, coef0=1.0):
    """Compute KID (mean, std) between two directories of images.

    Uses polynomial kernel and averages unbiased MMD^2 over `subsets` random
    pairs of subsets of size `subset_size`.
    """
    p1, p2 = paths
    if any(str(p).lower().endswith('.npz') for p in paths):
        raise ValueError('KID requires image directories, not .npz stats files.')
    if not os.path.isdir(p1) or not os.path.isdir(p2):
        raise RuntimeError('KID expects two directories of images.')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    act1 = _compute_activations_of_path(p1, model, batch_size, dims, device)
    act2 = _compute_activations_of_path(p2, model, batch_size, dims, device)

    m = min(len(act1), len(act2), subset_size)
    if m < 2:
        raise ValueError('Not enough samples for KID computation.')

    rng = np.random.default_rng()
    scores = []
    for _ in range(subsets):
        idx1 = rng.choice(len(act1), size=m, replace=False)
        idx2 = rng.choice(len(act2), size=m, replace=False)
        a = act1[idx1]
        b = act2[idx2]
        k_xx = _poly_kernel(a, a, degree=degree, gamma=gamma, coef0=coef0)
        k_yy = _poly_kernel(b, b, degree=degree, gamma=gamma, coef0=coef0)
        k_xy = _poly_kernel(a, b, degree=degree, gamma=gamma, coef0=coef0)
        scores.append(_mmd2_unbiased(k_xx, k_yy, k_xy))

    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean(scores)), float(np.std(scores))


from typing import Optional


def _resolve_device(device_str: Optional[str]):
    if device_str in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        return torch.device(device_str)
    except Exception as e:
        raise ValueError(f'Invalid device string: {device_str}') from e


def main():
    parser = ArgumentParser(description='Compute FID and KID between two paths')
    parser.add_argument('paths', type=str, nargs=2,
                        help='Two paths: image directories or .npz stats (for FID)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for Inception forward pass')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help='Inception feature dimensionality (default: 2048)')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device: 'auto', 'cpu', 'cuda', or 'cuda:0'")
    parser.add_argument('--no-kid', action='store_true',
                        help='Skip KID computation')
    parser.add_argument('--kid-subset-size', type=int, default=1000,
                        help='Subset size for KID')
    parser.add_argument('--kid-subsets', type=int, default=50,
                        help='Number of subsets for KID')
    parser.add_argument('--kid-degree', type=int, default=3,
                        help='Polynomial kernel degree for KID')
    parser.add_argument('--kid-coef0', type=float, default=1.0,
                        help='Polynomial kernel coef0 for KID')
    parser.add_argument('--kid-gamma', type=float, default=None,
                        help='Polynomial kernel gamma for KID (default: 1/d)')

    args = parser.parse_args()

    device = _resolve_device(args.device)

    # FID (works for dirs or .npz stats)
    fid_value = calculate_fid_given_paths(args.paths,
                                          batch_size=args.batch_size,
                                          device=device,
                                          dims=args.dims)
    print(f'FID: {fid_value:.6f}')

    # KID (requires dirs of images)
    if not args.no_kid:
        try:
            kid_mean, kid_std = calculate_kid_given_paths(
                args.paths,
                batch_size=args.batch_size,
                device=device,
                dims=args.dims,
                subset_size=args.kid_subset_size,
                subsets=args.kid_subsets,
                degree=args.kid_degree,
                gamma=args.kid_gamma,
                coef0=args.kid_coef0,
            )
            print(f'KID: {kid_mean:.6f} (+/- {kid_std:.6f})')
        except Exception as e:
            print(f'Skipping KID: {e}')


if __name__ == '__main__':
    main()
