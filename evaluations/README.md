Evaluation (FID/KID)
====================

This package provides a minimal, robust CLI to compute FID and KID between two
image sets using an InceptionV3 backbone compatible with the official FID
implementation.

Quick start
-----------

- Compute FID and KID:
  - `python -m evaluations.fid_score <fake_dir> <real_dir>`
- Supported inputs:
  - Directories of images (jpg, jpeg, png, bmp, tif, tiff) for both metrics
  - Or `.npz` stats (mu, sigma) for FID only (KID requires images)

Options
-------

- `--device auto|cpu|cuda|cuda:0` (default: `auto`)
- `--batch-size N` (default: 64)
- `--dims {64,192,768,2048}` (default: 2048)
- `--no-kid` to skip KID computation
- `--kid-subset-size N` (default: 1000)
- `--kid-subsets N` (default: 50)
- `--kid-degree D` (default: 3)
- `--kid-coef0 C` (default: 1.0)
- `--kid-gamma G` (default: 1/d)

Examples
--------

- Compute both metrics on default device:
  - `python -m evaluations.fid_score ./results/<name>/test_latest/images/fake_B ./datasets/<data>/testB`
- Force CPU and larger batch size:
  - `python -m evaluations.fid_score <fake_dir> <real_dir> --device cpu --batch-size 128`
- FID using cached stats for reals (no KID):
  - `python -m evaluations.fid_score <fake_dir> ./stats/real_stats.npz --no-kid`

Notes
-----

- KID uses an unbiased MMD^2 estimate with a polynomial kernel and reports the
  mean ± std across subsets.
- FID and KID are computed on the same Inception features (default: 2048-dim).
- If you need bitwise-reproducible results, fix seeds at the caller level and
  avoid GPU nondeterministic kernels.

