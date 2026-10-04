"""
GPU 优化训练配置 - RTX 4060 (8GB)
====================================
在 Jupyter Notebook 中运行此单元格以获取优化的训练配置

硬件配置:
- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM)
- CPU: Intel i9-13900HX
- 推荐模型: YOLO11s 或 YOLO11m
"""

import torch

print("=" * 80)
print("🔍 GPU 环境检测")
print("=" * 80)

if torch.cuda.is_available():
    print(f"✅ CUDA 可用")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   CUDA 版本: {torch.version.cuda}")
    device = '0'
else:
    print("❌ CUDA 不可用，将使用 CPU（速度较慢）")
    print("   建议安装 CUDA 版 PyTorch:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    device = 'cpu'

print("=" * 80)

# ============================================================
# 方案 A: 快速测试版（推荐首次运行）
# 预计时间: 30-40 分钟
# ============================================================
train_config_fast = {
    'data': 'data/GlobalWheat2020/wheat.yaml',
    'model': 'yolo11s.pt',      # 小模型，速度快
    'epochs': 30,                # 减少轮数
    'batch': 8,                  # RTX 4060 完全可承受
    'imgsz': 640,                # 降低分辨率，大幅加速
    'device': device,
    'project': 'runs/train',
    'name': 'yolo11s_wheat_fast',
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'patience': 10,              # 早停：10轮无提升则停止
    'cache': 'ram',              # 内存缓存，加速数据加载
    'amp': True,                 # 混合精度训练
    'plots': True,
    'verbose': True,
    'workers': 8,                # 数据加载线程数
}

# ============================================================
# 方案 B: 标准训练版（生产环境）
# 预计时间: 2-3 小时
# ============================================================
train_config_standard = {
    'data': 'data/GlobalWheat2020/wheat.yaml',
    'model': 'yolo11s.pt',       # 或改为 yolo11m.pt（更准确但更慢）
    'epochs': 50,                # 完整训练
    'batch': 8,                  # 平衡速度和显存
    'imgsz': 1024,               # 高分辨率，精度更高
    'device': device,
    'project': 'runs/train',
    'name': 'yolo11s_wheat_standard',
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'patience': 15,              # 更多耐心
    'cache': 'ram',              # 内存缓存
    'amp': True,
    'plots': True,
    'verbose': True,
    'workers': 8,
}

# ============================================================
# 方案 C: 高精度版（追求最佳性能）
# 预计时间: 4-6 小时
# ============================================================
train_config_high_accuracy = {
    'data': 'data/GlobalWheat2020/wheat.yaml',
    'model': 'yolo11m.pt',       # 中等模型，精度更高
    'epochs': 100,               # 更多轮数
    'batch': 4,                  # 减小 batch 以适应大模型
    'imgsz': 1024,               # 高分辨率
    'device': device,
    'project': 'runs/train',
    'name': 'yolo11m_wheat_hq',
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5.0,
    'patience': 20,
    'cache': 'ram',
    'amp': True,
    'plots': True,
    'verbose': True,
    'workers': 8,
}

# 选择要使用的配置（默认使用快速版）
# 修改这里来选择不同的配置：
# train_config = train_config_fast         # 快速测试
# train_config = train_config_standard     # 标准训练
# train_config = train_config_high_accuracy # 高精度
train_config = train_config_fast

print("\n" + "=" * 80)
print("⚙️  训练配置（GPU 优化版）")
print("=" * 80)
for key, value in train_config.items():
    print(f"  {key}: {value}")
print("=" * 80)

# 估算训练时间
if device != 'cpu':
    imgsz = train_config['imgsz']
    epochs = train_config['epochs']
    
    if imgsz == 640:
        time_per_epoch_min = 1.5
        time_per_epoch_max = 2.5
    elif imgsz == 1024:
        time_per_epoch_min = 3
        time_per_epoch_max = 5
    else:
        time_per_epoch_min = 2
        time_per_epoch_max = 4
    
    total_min = int(epochs * time_per_epoch_min)
    total_max = int(epochs * time_per_epoch_max)
    
    print(f"\n⏱️  预计训练时间:")
    print(f"   每 epoch: {time_per_epoch_min}-{time_per_epoch_max} 分钟")
    print(f"   总计 ({epochs} epochs): {total_min}-{total_max} 分钟 ({total_min//60}-{total_max//60} 小时)")
    print(f"   配置方案: {'快速测试' if train_config == train_config_fast else '标准训练' if train_config == train_config_standard else '高精度'}")
else:
    print("\n⚠️  使用 CPU 训练，速度会非常慢！")
    print("   强烈建议安装 CUDA 版 PyTorch")

print("\n💡 提示:")
print("   - 如需切换配置，修改上面的 train_config 赋值语句")
print("   - 快速版适合验证流程，标准版适合正式训练")
print("   - 如果显存不足，减小 batch 或 imgsz")
