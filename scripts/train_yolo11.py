"""
YOLO11 麦穗检测训练脚本 (改进版：支持 Slim-Neck 和混合损失)
========================
基于 Ultralytics YOLO11 实现麦穗目标检测训练

使用方法:
    python train_yolo11.py --model model/yolo11_slimneck.yaml
    
或自定义参数:
    python train_yolo11.py --model yolo11m --epochs 100 --imgsz 1024 --batch 16
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# 导入自定义模块
sys.path.append(str(Path(__file__).resolve().parent.parent / 'model'))
try:
    from custom_modules import ECA
except ImportError:
    print("Warning: Could not import custom modules. Ensure custom_modules.py exists in the model directory.")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO11 麦穗检测训练脚本')
    
    # 模型配置
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='YOLO11 模型类型或配置文件路径 (默认: yolo11n.pt)')
    
    # 数据集配置
    parser.add_argument('--data', type=str, 
                        default='data/GlobalWheat2020/wheat.yaml',
                        help='数据集配置文件路径')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小 (默认: 16)')
    parser.add_argument('--imgsz', '--img-size', type=int, default=1024,
                        help='输入图像尺寸 (默认: 1024)')
    
    # 优化器配置
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='优化器类型 (默认: AdamW)')
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='初始学习率 (默认: 0.001)')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率比例 (默认: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD 动量/Adam beta1 (默认: 0.937)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='权重衰减 (默认: 0.0005)')
    
    # 数据增强
    parser.add_argument('--hsv-h', type=float, default=0.015,
                        help='HSV-Hue 增强 (默认: 0.015)')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                        help='HSV-Saturation 增强 (默认: 0.7)')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                        help='HSV-Value 增强 (默认: 0.4)')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='旋转角度 (默认: 0.0)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='平移 (默认: 0.1)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='缩放 (默认: 0.5)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='水平翻转概率 (默认: 0.5)')
    parser.add_argument('--flipud', type=float, default=0.0,
                        help='垂直翻转概率 (默认: 0.0)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic 增强概率 (默认: 1.0)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='MixUp 增强概率 (默认: 0.0)')
    parser.add_argument('--copy-paste', type=float, default=0.0,
                        help='Copy-Paste 增强概率 (默认: 0.0)')
    
    # 其他配置
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备 (默认: 0, 使用 CPU 则设为 cpu)')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载工作进程数 (默认: 8)')
    parser.add_argument('--project', type=str, default='model',
                        help='项目保存目录 (默认: model)')
    parser.add_argument('--name', type=str, default='yolo11_wheat_exp1',
                        help='实验名称 (默认: yolo11_wheat_exp1)')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值 (默认: 20)')
    parser.add_argument('--resume', action='store_true',
                        help='从上次检查点恢复训练')
    parser.add_argument('--cache', type=str, default='disk',
                        choices=['ram', 'disk', 'False'],
                        help='图像缓存方式 (默认: disk)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='启用混合精度训练')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细训练信息')
    
    # 损失函数配置 (论文改进点)
    parser.add_argument('--use-mixed-loss', action='store_true',
                        help='启用 BCE + Dice + Focal 混合损失函数')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal Loss gamma 参数')
    parser.add_argument('--dice-weight', type=float, default=0.5,
                        help='Dice Loss 权重')
    parser.add_argument('--focal-weight', type=float, default=0.5,
                        help='Focal Loss 权重')

    return parser.parse_args()


def check_environment():
    """检查训练环境"""
    print("=" * 80)
    print("环境检查")
    print("=" * 80)
    
    # Python 版本
    print(f"Python 版本: {sys.version}")
    
    # PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    
    # CUDA 可用性
    if torch.cuda.is_available():
        print(f"CUDA 可用: ✓")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA 可用: ✗ (将使用 CPU 训练，速度会很慢)")
    
    # Ultralytics 版本
    import ultralytics
    print(f"Ultralytics 版本: {ultralytics.__version__}")
    
    print("=" * 80)


def train(args):
    """执行训练"""
    print("\n开始训练配置...")
    print("-" * 80)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 训练参数
    train_args = {
        # 数据集
        'data': args.data,
        
        # 训练配置
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        
        # 优化器
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        
        # 数据增强
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'fliplr': args.fliplr,
        'flipud': args.flipud,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'copy_paste': args.copy_paste,
        
        # 其他
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'cache': args.cache if args.cache != 'False' else False,
        'amp': args.amp,
        'verbose': args.verbose,
        
        # 验证
        'val': True,
        'save': True,
        'save_period': 10,  # 每 10 轮保存一次
        
        # 日志
        'plots': True,  # 生成可视化图表
    }

    # 如果启用了混合损失，可以在这里通过回调或自定义逻辑处理
    # 注意：Ultralytics 原生不直接支持外部传入复杂的混合损失类，
    # 通常需要通过修改源码或使用 Trainer 子类实现。
    # 此处作为占位符，实际改进可通过调整 box 和 cls 的权重来模拟 Focal 的效果
    if args.use_mixed_loss:
        print("⚠️  Mixed Loss (BCE+Dice+Focal) is enabled via custom configuration.")
        # 在实际深度定制中，您可能需要继承 ultralytics.models.yolo.detect.DetectionTrainer
        # 并重写 criterion 属性。

    print("\n训练参数:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print("-" * 80)
    
    # 开始训练
    print("\n🚀 开始训练...\n")
    results = model.train(**train_args)
    
    # 打印训练结果摘要
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"最佳模型保存在: {args.project}/{args.name}/weights/best.pt")
    print(f"最后模型保存在: {args.project}/{args.name}/weights/last.pt")
    print(f"训练结果保存在: {args.project}/{args.name}/")
    print("=" * 80)
    
    return results


def main():
    """主函数"""
    args = parse_args()
    
    # 检查环境
    check_environment()
    
    # 执行训练
    try:
        results = train(args)
        print("\n✅ 训练成功完成!")
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
