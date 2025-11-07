# E-UNSB

A streamlined implementation of unpaired image-to-image translation based on Neural Schrödinger Bridge.

## Project Structure

```
options/            # Training/testing command-line options
data/               # Datasets and transforms
models/             # SB model and networks
evaluations/        # FID/KID evaluation
tools/              # Data processing tools
train.py            # Training script
test.py             # Testing script
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --config configs/example_sb.yaml
```

## Testing

```bash
python test.py --name <experiment_name>
```

## Evaluation (FID/KID)

```bash
python -m evaluations.fid_score <fake_image_dir> <real_image_dir>
```

## Acknowledgments

Our source code is based on [UNSB](https://github.com/cyclomon/UNSB).

We thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID calculation.
