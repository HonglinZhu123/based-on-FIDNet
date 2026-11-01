# FIDNet：幅度谱多尺度空洞卷积实验

基于 [FIDNet 官方实现](https://github.com/MIV-XJTU/FIDNet)，仅替换 APB
（`Amplitude_Phase_Block`）中的幅度谱卷积，保留原版用于对照。
这是结构实验，尚未训练，也未证明 PSNR、SSIM 或 SAM 有改善。

## 模型文件

- `model/FIDNet.py`：原版网络，来自上游提交 `cde4e2a`，内容保持不变。
- `model/FIDNet_MS.py`：新模型 `FIDNetMS`，继承原版并替换两个 APB 的 `amp_fuse_1`。
- `docs/FIDNet_UPSTREAM_README.md`：原版说明及论文引用。
- `tests/test_fidnet_ms.py`：修改范围、尺寸、完整网络梯度及权重重载检查。

## 修改内容

原版幅度修正量由一个普通 `3×3` 卷积产生；新版本使用：

```text
幅度谱 [B×波段数, C, H, W//2+1]
  → 1×1 通道映射
  → 三路并行 3×3 深度空洞卷积，实际 dilation = 1、2、3
  → 通道拼接 → 1×1 融合 → 幅度修正量
  → 与原幅度谱相加（沿用 APB 原有的外部残差）
```

多尺度深度空洞卷积和拼接融合的结构参考了 HDST
（[Hybrid-Domain Synergistic Transformer for Hyperspectral Image Denoising](https://arxiv.org/abs/2507.20099)）
本地实现 `HDST-HSIDenoise/models/sert.py` 的 `ASPP_Enhanced`，这里是针对 APB 的改写，
并非完整移植该模块。

- `padding=dilation`，不改变频谱特征的高、宽；三个分支的有效覆盖范围为 `3×3、5×5、7×7`。
- 膨胀率直接采用传入值，不使用 HDST 实现中的额外 `×2`。
- 不增加 BN、激活、Dropout、全局池化或内部残差，以缩小本轮实验范围。
- 相位卷积、FFT/IFFT、APB 前后卷积、空频融合及光谱网络均沿用原版。
- 这里的多尺度操作发生在频谱坐标上，不能直接等同于空间图像的多尺度邻域。

## 安装与使用

验证环境：Python 3.10、PyTorch 2.3.1、einops 0.8.1。

```bash
pip install -r requirements.txt
```

在仓库根目录运行：

```python
import torch
from model.FIDNet import FIDNet       # 原版对照
from model.FIDNet_MS import FIDNetMS   # 幅度谱多尺度版本

# 下列宽度仅用于运行示例；正式实验应与自己的基线配置一致。
model = FIDNetMS(hid_S=8, hid_T=2, dilation_rates=(1, 2, 3))
x = torch.randn(1, 31, 32, 32)  # [批次, 光谱波段, 高, 宽]
y = model(x)
assert y.shape == x.shape
```

沿用原版结构约束：`hid_S` 应为 4 的倍数，图像高、宽应能被 4 整除。
APB 单独支持奇数尺寸，但完整网络的原版 3D U-Net 跳跃相加有上述尺寸约束。

## 验证

```bash
python -m unittest discover -s tests -v
```

检查只有幅度分支参数发生结构变化，APB 的奇偶尺寸输出与有限梯度，
完整网络在 CPU/CUDA 上的前向、反向及 `state_dict` 重载。
CUDA 不可用时跳过对应检查。测试使用小规模随机输入，不代表训练或去噪质量评估。

## 训练与权重

本仓库不包含数据集、预训练权重或完整训练管线。原版训练/测试接入方法见
[上游说明](docs/FIDNet_UPSTREAM_README.md)，可使用其引用的
[RAS2S](https://github.com/MIV-XJTU/RAS2S) 训练框架；接入时应导入 `FIDNetMS`，
并将两个模型文件放在同一个 Python 包内。

新分支参数与原版 `fidnet.pth` 不兼容，不能将原版权重直接严格加载为新模型。
建议先从头训练并与同配置的原版对比；若部分加载旧权重，需排除所有
`amp_fuse_1` 键并检查缺失键，不能把未训练的新分支当成有效预训练模型。
新版本保存的权重应使用 `FIDNetMS` 重新构造后加载。

## 来源与引用

原版网络作者为 Jiahua Xiao 等人，论文为
*Bridging Fourier and Spatial-Spectral Domains for Hyperspectral Image Denoising*，ACM MM 2024。
本仓库为基于该实现的实验版本；原始引用信息保留于
[上游 README](docs/FIDNet_UPSTREAM_README.md)。HDST 的结构参考见上文。
