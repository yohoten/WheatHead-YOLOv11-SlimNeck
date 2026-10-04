"""
CPU 训练优化配置 - 基于当前环境
====================================
针对 PyTorch CPU 版本 + i9-13900HX 的优化配置
"""

import torch

print("=" * 80)
print("🔍 环境检测")
print("=" * 80)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print("=" * 80)

# ============================================================
# CPU 优化配置（推荐使用）
# ============================================================
train_config = {
    'data': 'data/GlobalWheat2020/wheat.yaml',
    'model': 'yolo11n.pt',       # ✅ 最小模型，速度最快
    'epochs': 30,                # ✅ 配合早停机制
    'batch': 4,                  # ✅ CPU 下的平衡点
    'imgsz': 640,                # ✅ 低分辨率加速训练
    'device': 'cpu',             # ✅ 强制使用 CPU
    'project': 'runs/train',
    'name': 'yolo11n_wheat_cpu_optimized',
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'patience': 10,              # ✅ 10轮无提升则早停
    'cache': 'disk',             # ✅ CPU 下更稳定
    'amp': False,                # 🔴 CPU 不支持混合精度
    'plots': True,
    'verbose': True,
    'workers': 4,                # 🟡 减少数据加载线程
}

print("\n" + "=" * 80)
print("⚙️  CPU 优化训练配置")
print("=" * 80)
for key, value in train_config.items():
    print(f"  {key}: {value}")
print("=" * 80)

# 估算训练时间
print(f"\n⏱️  预计训练时间:")
print(f"   每 epoch: ~15-20 分钟（i9-13900HX CPU + yolo11n + imgsz=640）")
print(f"   总计 ({train_config['epochs']} epochs): ~{30*15//60}-{30*20//60} 小时")
print(f"   实际可能更早停止（patience={train_config['patience']}）")
print()
print("💡 提示:")
print("   1. 这是临时方案，用于验证流程和数据")
print("   2. CUDA PyTorch 安装完成后，请切换到 GPU 配置")
print("   3. GPU 训练将提速 50-100 倍（~45 分钟完成）")
print("   4. 检查 CUDA 状态: check_cuda_status.bat")
