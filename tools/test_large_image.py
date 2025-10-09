import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image,ImageOps
from torchvision import transforms
from options.test_options import TestOptions
from models import create_model
from util.visualizer import save_images
from util import _html_
import util.util as util


class LargeImageTester:
    def __init__(self, opt):
        self.opt = opt
        self.model = create_model(opt)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型并尽量迁移到同一设备
        self.model.setup(opt)
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @staticmethod
    def _tile_positions(total, tile, overlap):
        """
        生成每一维的起点索引列表，保证覆盖到最后 (total - tile)，
        并避免出现空切片或非 tile 大小的块。
        """
        s = tile - overlap
        if total <= tile:
            return [0]
        pos = list(range(0, max(total - tile, 0) + 1, s))
        if pos[-1] != total - tile:
            pos.append(total - tile)
        return pos

    def process_tile(self, tile_A, tile_B):
        """处理单个 tile 对，期望输入 [3,256,256]，输出 [1,3,256,256]"""
        data = {
            'A': tile_A.unsqueeze(0).to(self.device),  # [1,3,256,256]
            'B': tile_B.unsqueeze(0).to(self.device),
            'A_paths': [''],
            'B_paths': ['']
        }
        data2 = {
            'A': tile_A.unsqueeze(0).to(self.device),
            'B': tile_B.unsqueeze(0).to(self.device),
            'A_paths': [''],
            'B_paths': ['']
        }

        # 注意：你的工程原先就是 set_input(data, data2)
        # 如果你的模型其实只需要一个 data，请改为 self.model.set_input(data)
        self.model.set_input(data, data2)

        with torch.no_grad():
            self.model.test()

        out = self.model.get_current_visuals().get('fake_1')
        # 统一为 GPU float32 4D tensor
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out)
        if isinstance(out, Image.Image):
            out = self.transform(out)  # -> [3,H,W]
        if torch.is_tensor(out):
            if out.dim() == 3:
                out = out.unsqueeze(0)  # -> [1,3,H,W]
            out = out.to(self.device).float()
        else:
            raise TypeError("model.get_current_visuals()['fake_B'] 返回类型不受支持")
        return out  # [1,3,256,256]

    def test_large_image_pair(self, image_A_path, image_B_path, tile_size=256, overlap=32):
        """处理大尺寸图像对，所有 tile 恒为 256x256"""
        # 读取A域和B域的图像
        img_A = Image.open(image_A_path).convert('RGB')
        img_A = ImageOps.invert(img_A)
        img_B = Image.open(image_B_path).convert('RGB')
        w, h = img_A.size

        print(f"处理图像对:\nA: {image_A_path}\nB: {image_B_path}\n尺寸: {w}x{h}")

        # 预处理输入图像到张量（先不裁剪）
        img_A_tensor = self.transform(img_A).to(self.device)  # [3,H,W]
        img_B_tensor = self.transform(img_B).to(self.device)

        # 若尺寸不同，先把 B resize 到 A 的尺寸（防止切片错位）
        if img_B.size != img_A.size:
            img_B = img_B.resize(img_A.size, Image.BICUBIC)
            img_B_tensor = self.transform(img_B).to(self.device)

        # 若任一维小于 tile_size，复制边界填充到至少 tile_size
        pad_h = max(0, tile_size - img_A_tensor.shape[1])
        pad_w = max(0, tile_size - img_A_tensor.shape[2])
        if pad_h > 0 or pad_w > 0:
            img_A_tensor = F.pad(img_A_tensor, (0, pad_w, 0, pad_h), mode='replicate')
            img_B_tensor = F.pad(img_B_tensor, (0, pad_w, 0, pad_h), mode='replicate')

        H, W = img_A_tensor.shape[1], img_A_tensor.shape[2]

        # 输出/计数缓存基于(可能填充后的)尺寸
        output = torch.zeros((1, 3, H, W), device=self.device, dtype=torch.float32)
        count = torch.zeros_like(output)

        # 计算 tile 起点，确保最后一块起点为 H - tile/W - tile
        tops = self._tile_positions(H, tile_size, overlap)
        lefts = self._tile_positions(W, tile_size, overlap)
        print(f"将分成 {len(tops)}x{len(lefts)} 个 256x256 块进行处理")

        for i, top0 in enumerate(tops):
            for j, left0 in enumerate(lefts):
                # 钳制起点，避免越界或空切片
                top = int(min(max(top0, 0), max(H - tile_size, 0)))
                left = int(min(max(left0, 0), max(W - tile_size, 0)))
                bottom = top + tile_size
                right = left + tile_size

                print(f"处理块 ({i},{j}): 位置 [{top}:{bottom}, {left}:{right}]")

                tile_A = img_A_tensor[:, top:bottom, left:right]  # [3,256,256]
                tile_B = img_B_tensor[:, top:bottom, left:right]

                # 兜底：任何维度不是 256（含极端边缘情况），复制边界填充到 256
                if tile_A.shape[1] != tile_size or tile_A.shape[2] != tile_size:
                    pad_h2 = max(0, tile_size - tile_A.shape[1])
                    pad_w2 = max(0, tile_size - tile_A.shape[2])
                    if pad_h2 > 0 or pad_w2 > 0:
                        tile_A = F.pad(tile_A, (0, pad_w2, 0, pad_h2), mode='replicate')
                        tile_B = F.pad(tile_B, (0, pad_w2, 0, pad_h2), mode='replicate')

                with torch.no_grad():
                    result = self.process_tile(tile_A, tile_B)  # [1,3,256,256]

                # 累加（简单平均；如需更平滑可改为Hann窗权重）
                output[:, :, top:bottom, left:right] += result
                count[:, :, top:bottom, left:right] += 1

        # 平均并裁回原始尺寸
        output = (output / count.clamp_min(1e-6)).clamp(-1, 1)
        output = output[:, :, :h, :w]
        return output


def main():
    # 获取测试配置
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # 创建保存目录
    save_dir = os.path.join(opt.results_dir, 'AF2HE_results')
    os.makedirs(save_dir, exist_ok=True)

    # 创建网页用于查看结果
    web_dir = os.path.join(save_dir, opt.name)
    webpage = _html_.HTML(web_dir, f'Experiment = {opt.name}, Phase = test')

    # 初始化测试器
    tester = LargeImageTester(opt)

    # 获取测试图像路径对
    testA_dir = os.path.join(opt.dataroot, 'testA')
    testB_dir = os.path.join(opt.dataroot, 'testB')
    image_files_A = sorted([f for f in os.listdir(testA_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))])
    image_files_B = sorted([f for f in os.listdir(testB_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))])

    # 处理每对图像
    for img_A_name, img_B_name in zip(image_files_A, image_files_B):
        img_A_path = os.path.join(testA_dir, img_A_name)
        img_B_path = os.path.join(testB_dir, img_B_name)

        print(f"\n开始处理图像对: {img_A_name} - {img_B_name}")

        # 生成输出文件路径
        save_name = os.path.splitext(img_A_name)[0]

        # 处理图像
        output = tester.test_large_image_pair(
            img_A_path,
            img_B_path,
            tile_size=256,
            overlap=32
        )

        # 保存拼接结果到文件
        # stitched_dir = os.path.join(save_dir, "stitched")
        # os.makedirs(stitched_dir, exist_ok=True)

        # stitched_np = util.tensor2im(output)  # HxWx3, uint8
        # stitched_path = os.path.join(stitched_dir, f"{save_name}_stitched.png")
        # util.save_image(stitched_np, stitched_path)
        # print("Stitched image saved to:", stitched_path)

        # 保存到网页（键名改成更常见的 fake_B）
        visuals = {'fake_1': util.tensor2im(output)}
        save_images(webpage, visuals, [img_A_path], aspect_ratio=1.0, width=opt.display_winsize)

    webpage.save()


if __name__ == '__main__':
    main()
