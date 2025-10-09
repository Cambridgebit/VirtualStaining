UNSB SB Minimal（PyTorch）
=========================

该仓库是 UNSB（Unpaired Neural Schrödinger Bridge）的精简可运行子集，面向非配对图像到图像翻译。包含核心训练/测试流程、数据加载、模型与常用工具（大图切片/拼接、FID/KID 评估、大图推理等）。

快速开始（训练）
---------------

- 直接启动（零参数，自动 GPU/数据集/实验名，默认关闭 Visdom）：

```
python train.py
```

- 使用配置文件（推荐）：

```
python train.py --config configs/example_sb.yaml
```

- 在配置之上临时覆盖：

```
python train.py --config configs/example_sb.yaml --lr 0.0001 --gpu_ids 0
```

默认行为说明
------------

- `--gpu_ids`：默认 `auto`（有 CUDA 则用 GPU:0，否则用 CPU）。
- `--dataroot`：若未设置或无效，自动在 `./datasets` 中选择包含 `trainA`/`trainB` 的首个目录。
- `--name`：若保持默认 `experiment_name`，自动命名为 `<dataset>_<model>`。
- Visdom：默认关闭（`--display_id -1`）。如需可视化，传 `--display_id 1` 并手动启动 `python -m visdom.server -p 8084`。

测试
----

```
python test.py
```

脚本会加载 checkpoint 并按固定设置（单线程、batch=1、顺序取样、无翻转）保存 HTML 结果到 `./results/<name>/<phase>_<epoch>/`。

评估（FID/KID & PRDC）
---------------------

- FID 与 KID（推荐方式）：

```
python -m evaluations.fid_score <fake_dir> <real_dir>
```

说明：
- 支持图片目录（jpg/jpeg/png/bmp/tif/tiff）。
- FID 也支持 `.npz` 统计（mu/sigma）；KID 仅支持图片目录。
- 选项：`--device auto|cpu|cuda|cuda:0`（默认 auto）、`--batch-size 64`、`--dims 2048`、`--no-kid`、`--kid-subset-size 1000`、`--kid-subsets 50` 等。

- PRDC：

```
python -m evaluations.DC <real_dir> <fake_dir>
```

工具去重/兼容性
---------------

- 旧脚本 `tools/caculate_FID_KID.py` 已标记为“已弃用（DEPRECATED）”，并作为兼容入口内部转调 `evaluations.fid_score`。建议统一使用：
  - `python -m evaluations.fid_score <fake_dir> <real_dir>`

大图推理
--------

```
python tools/test_large_image.py \
  --dataroot ./datasets/<your_dataset> \
  --name example_SB --checkpoints_dir ./checkpoints \
  --mode sb --eval --gpu_ids 0
```

目录结构
--------

```
options/            # 训练/测试选项
data/               # 数据集与变换
models/             # 模型与网络
util/               # 工具（可视化、HTML、杂项）
evaluations/        # 评估（FID/KID、PRDC）
tools/              # 数据/大图脚本与兼容入口
configs/            # 配置样例
datasets/           # 数据集根目录（包含 trainA/trainB/testA/testB）
results/            # 结果输出
train.py            # 训练入口
test.py             # 测试入口（精简）
```

依赖环境
--------

- 核心：`torch`、`torchvision`、`torchaudio`、`numpy`、`pillow`、`tqdm`
- 可视化/HTML：`visdom`、`dominate`
- 数据/工具：`opencv-python`、`tifffile`、`requests`、`beautifulsoup4`、`lxml`、`scipy`、`scikit-learn`、`packaging`、`GPUtil`
- 评估（可选）：`pytorch-fid`

英文文档
--------

请参阅 `README_EN.md` 以获得完整英文说明（训练、测试与评估的简化用法与默认行为）。

