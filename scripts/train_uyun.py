"""
优云智算平台 - YOLO11 麦穗检测训练脚本
========================================
适配优云智算云平台的训练配置，支持：
- GPU 自动检测和使用
- 分布式训练（多卡）
- 断点续训
- 结果自动保存

使用方法:
    python train_uyun.py
    
或自定义参数:
    python train_uyun.py --model yolo11m --epochs 100 --batch 32 --device 0,1
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def check_environment():
    """检查运行环境"""
    print("=" * 80)
    print("优云智算平台 - 环境检查")
    print("=" * 80)
    
    # Python 版本
    print(f"Python 版本: {sys.version}")
    
    # PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    
    # CUDA 信息
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"          显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("❌ CUDA 不可用，将使用 CPU 训练（速度较慢）")
    
    # 工作目录
    print(f"工作目录: {Path.cwd()}")
    print("=" * 80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='优云智算平台 - YOLO11 麦穗检测训练')
    
    # 模型配置
    parser.add_argument('--model', type=str, default='yolo11m.pt',
                        choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 
                                 'yolo11l.pt', 'yolo11x.pt'],
                        help='YOLO11 模型类型 (默认: yolo11m.pt)')
    
    # 数据集配置
    parser.add_argument('--data', type=str, 
                        default='data/GlobalWheat2020/wheat.yaml',
                        help='数据集配置文件路径')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--batch', type=int, default=32,
                        help='批次大小 (默认: 32，GPU环境下可增大)')
    parser.add_argument('--imgsz', '--img-size', type=int, default=1024,
                        help='输入图像尺寸 (默认: 1024)')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备 (默认: 0; 多卡: 0,1,2,3; CPU: cpu)')
    
    # 优化器配置
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='优化器类型 (默认: AdamW)')
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='初始学习率 (默认: 0.001)')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率比例 (默认: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='动量 (默认: 0.937)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='权重衰减 (默认: 0.0005)')
    
    # 训练策略
    parser.add_argument('--warmup-epochs', type=float, default=3.0,
                        help='预热轮数 (默认: 3.0)')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值 (默认: 20)')
    parser.add_argument('--cache', type=str, default='ram',
                        choices=['ram', 'disk', 'False'],
                        help='数据缓存方式 (默认: ram，加速训练)')
    
    # 输出配置
    parser.add_argument('--project', type=str, default='runs/train',
                        help='项目输出目录 (默认: runs/train)')
    parser.add_argument('--name', type=str, default='wheat_yolo11',
                        help='实验名称 (默认: wheat_yolo11)')
    parser.add_argument('--exist-ok', action='store_true',
                        help='如果项目目录存在则覆盖')
    
    # 其他配置
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载工作线程数 (默认: 8)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用预训练权重')
    
    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()
    
    # 环境检查
    check_environment()
    
    # 确定设备
    if args.device == 'cpu':
        device = 'cpu'
        print("\n⚠️  使用 CPU 训练")
    else:
        # 自动选择可用的 GPU
        if torch.cuda.is_available():
            device = args.device
            print(f"\n✅ 使用 GPU: {device}")
        else:
            device = 'cpu'
            print("\n⚠️  CUDA 不可用，切换到 CPU")
    
    # 创建输出目录
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir.absolute()}")
    
    # 加载模型
    print(f"\n🔄 加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 训练配置
    train_config = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': device,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'cache': args.cache if args.cache != 'False' else False,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'workers': args.workers,
        'pretrained': args.pretrained,
        'amp': True,  # 自动混合精度训练
        'plots': True,  # 生成训练图表
        'verbose': True,
        'save': True,
        'save_period': 10,  # 每10个epoch保存一次
    }
    
    # 打印训练配置
    print("\n" + "=" * 80)
    print("训练配置")
    print("=" * 80)
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # 开始训练
    print("\n🚀 开始训练...\n")
    try:
        results = model.train(**train_config)
        
        print("\n" + "=" * 80)
        print("✅ 训练完成!")
        print("=" * 80)
        print(f"模型保存位置: {output_dir / 'weights'}")
        print(f"最佳模型: {output_dir / 'weights' / 'best.pt'}")
        print(f"最后模型: {output_dir / 'weights' / 'last.pt'}")
        print(f"训练日志: {output_dir}")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
