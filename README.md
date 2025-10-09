UNSB SB Minimal (PyTorch)
=========================

A minimal, self-contained subset of the UNSB (Unpaired Neural Schrödinger Bridge) codebase for training and testing SB models on unpaired image-to-image translation tasks. It includes the core training/test pipelines, data loaders, models, and practical tools (data slicing/stitching, FID/KID evaluation, large-image inference).

Project Structure
----------------

```
options/            # CLI options (train/test)
data/               # datasets and transforms
models/             # SB model and networks
util/               # utilities (visualizer, html, misc)
evaluations/        # FID/KID (PyTorch implementation)
tools/              # data tools: slicing, stitching, large-image test
configs/            # example training configs
datasets/           # put your datasets here (trainA/trainB/testA/testB)
results/            # test outputs
train.py            # training entry
test_SB.py          # testing entry
```

Installation
------------

- Python 3.8+ recommended. Use CUDA-enabled PyTorch for GPU.
- Create an environment and install dependencies:

```
pip install -r requirements.txt
```

Training: Simple Defaults
-------------------------

- Minimal run: auto-detect GPU, dataset (first folder under `./datasets` containing `trainA`/`trainB`), and sensible experiment name.

```
python train.py
```

- Config-driven (recommended):

```
python train.py --config configs/example_sb.yaml
```

- Override config values on the command line:

```
python train.py --config configs/example_sb.yaml --lr 0.0001 --gpu_ids 0
```

Key Defaults and Behavior
-------------------------

- `--gpu_ids`: default `auto` (uses GPU 0 if available, else CPU).
- `--dataroot`: if not set or invalid, auto-picks a dataset under `./datasets` containing `trainA` or `trainB`.
- `--name`: if left as `experiment_name`, auto-named as `<dataset>_<model>`.
- Visdom disabled by default (`--display_id -1`). Enable with `--display_id 1` and run a Visdom server if needed.

Testing
-------

```
python test_SB.py \
  --dataroot ./datasets/example \
  --name example_SB \
  --checkpoints_dir ./checkpoints \
  --mode sb --eval --phase test --num_test 50 \
  --gpu_ids 0
```

Evaluation (FID/KID)
--------------------

- Compute FID and KID:

```
python -m evaluations.fid_score <fake_dir> <real_dir>
```

- Supported inputs:
  - Directories of images (jpg, jpeg, png, bmp, tif, tiff) for both metrics
  - Or `.npz` stats (mu, sigma) for FID only (KID requires images)

- Options:
  - `--device auto|cpu|cuda|cuda:0` (default: auto)
  - `--batch-size N` (default: 64)
  - `--dims {64,192,768,2048}` (default: 2048)
  - `--no-kid` to skip KID
  - `--kid-subset-size N` (default: 1000)
  - `--kid-subsets N` (default: 50)
  - `--kid-degree D` (default: 3), `--kid-coef0 C` (default: 1.0), `--kid-gamma G` (default: 1/d)

- Examples:
  - `python -m evaluations.fid_score ./results/<name>/test_latest/images/fake_B ./datasets/<data>/testB`
  - FID using cached stats for real images (no KID): `python -m evaluations.fid_score <fake_dir> ./stats/real_stats.npz --no-kid`

Large-Image Inference
---------------------

```
python tools/test_large_image.py \
  --dataroot ./datasets/<your_dataset> \
  --name example_SB --checkpoints_dir ./checkpoints \
  --mode sb --eval --gpu_ids 0
```

Requirements
------------

- Core: `torch`, `torchvision`, `torchaudio`, `numpy`, `pillow`, `tqdm`
- Visualization/HTML: `visdom`, `dominate`
- Data/tools: `opencv-python`, `tifffile`, `requests`, `beautifulsoup4`, `lxml`, `scipy`, `scikit-learn`, `packaging`, `GPUtil`
- Evaluation (optional): `pytorch-fid`

Citation
--------

If you use UNSB in your research:

```
@InProceedings{
  kim2023unsb,
  title={Unpaired Image-to-Image Translation via Neural Schrödinger Bridge},
  author={Beomsu Kim and Gihyun Kwon and Kwanyoung Kim and Jong Chul Ye},
  booktitle={ICLR},
  year={2024}
}
```

