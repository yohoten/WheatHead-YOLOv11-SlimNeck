"""
伪标签生成脚本
==============
使用训练好的模型对未标注数据生成伪标签，用于半监督学习

使用方法:
    python pseudo_labeling.py --model model/yolo11_wheat_exp1/weights/best.pt \
                              --unlabeled_data path/to/unlabeled/images \
                              --confidence_threshold 0.7
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='伪标签生成脚本')
    
    parser.add_argument('--model', type=str, 
                        default='model/yolo11_wheat_exp1/weights/best.pt',
                        help='训练好的模型路径')
    parser.add_argument('--unlabeled_data', type=str, required=True,
                        help='未标注图像目录')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='置信度阈值 (默认: 0.7)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='NMS IoU 阈值 (默认: 0.5)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='输入图像尺寸')
    parser.add_argument('--output_dir', type=str, 
                        default='data/GlobalWheat2020/pseudo_labels',
                        help='伪标签输出目录')
    parser.add_argument('--device', type=str, default='0',
                        help='设备')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化伪标签结果')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大处理样本数 (None 表示全部处理)')
    
    return parser.parse_args()


def convert_to_yolo_format(box, img_width, img_height):
    """
    将边界框转换为 YOLO 格式
    
    Args:
        box: [x1, y1, x2, y2] 像素坐标
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        [class_id, x_center, y_center, width, height] 归一化坐标
    """
    x1, y1, x2, y2 = box
    
    # 计算中心点和宽高
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # 确保在 [0, 1] 范围内
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width = np.clip(width, 0, 1)
    height = np.clip(height, 0, 1)
    
    return [0, x_center, y_center, width, height]  # class_id = 0 for wheat


def generate_pseudo_labels(args):
    """生成伪标签"""
    print("=" * 80)
    print("伪标签生成")
    print("=" * 80)
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 创建输出目录
    output_images_dir = os.path.join(args.output_dir, 'images')
    output_labels_dir = os.path.join(args.output_dir, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 获取未标注图像列表
    unlabeled_path = Path(args.unlabeled_data)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(unlabeled_path.glob(ext))
        image_files.extend(unlabeled_path.glob(ext.upper()))
    
    image_files = list(set(image_files))  # 去重
    
    if args.max_samples:
        image_files = image_files[:args.max_samples]
    
    print(f"找到 {len(image_files)} 张未标注图像")
    print(f"置信度阈值: {args.confidence_threshold}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 80)
    
    # 统计信息
    total_images = 0
    total_detections = 0
    skipped_images = 0
    
    # 处理每张图像
    for img_path in tqdm(image_files, desc="生成伪标签"):
        try:
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # 推理
            results = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.confidence_threshold,
                iou=args.iou_threshold,
                device=args.device,
                verbose=False,
                save=False,
            )
            
            result = results[0]
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                skipped_images += 1
                continue
            
            # 生成 YOLO 格式标签
            label_lines = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # 转换为 YOLO 格式
                yolo_box = convert_to_yolo_format([x1, y1, x2, y2], img_width, img_height)
                label_lines.append(f"{yolo_box[0]} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}")
                
                total_detections += 1
            
            # 保存标签文件
            img_name = img_path.stem
            label_file = os.path.join(output_labels_dir, f"{img_name}.txt")
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(label_lines))
            
            # 复制图像到输出目录
            output_img_path = os.path.join(output_images_dir, img_path.name)
            shutil.copy2(img_path, output_img_path)
            
            total_images += 1
            
            # 可视化 (可选)
            if args.visualize and total_images <= 10:
                visualize_prediction(img, boxes, img_name, args.output_dir)
        
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {str(e)}")
            continue
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("伪标签生成完成!")
    print("=" * 80)
    print(f"总处理图像数: {len(image_files)}")
    print(f"成功生成标签: {total_images}")
    print(f"跳过图像 (无检测): {skipped_images}")
    print(f"总检测数: {total_detections}")
    print(f"平均每张图像检测数: {total_detections / total_images if total_images > 0 else 0:.2f}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)
    
    # 生成使用说明
    generate_usage_guide(args.output_dir, total_images)


def visualize_prediction(img, boxes, img_name, output_dir):
    """可视化预测结果"""
    viz_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 绘制边界框
    img_viz = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_viz, f'{conf:.2f}', (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存可视化结果
    output_path = os.path.join(viz_dir, f"{img_name}_pseudo.jpg")
    cv2.imwrite(output_path, img_viz)


def generate_usage_guide(output_dir, num_pseudo_labels):
    """生成使用说明"""
    guide_file = os.path.join(output_dir, 'README_PSEUDO_LABELS.md')
    
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(f"""# 伪标签数据集说明

## 概述
本目录包含使用预训练 YOLO11 模型生成的伪标签数据。

## 数据统计
- 伪标签图像数量: {num_pseudo_labels}
- 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 目录结构
```
{os.path.basename(output_dir)}/
├── images/          # 图像文件
├── labels/          # YOLO 格式标签
└── visualization/   # 可视化示例 (前10张)
```

## 如何使用

### 方法 1: 合并到现有数据集
将伪标签数据合并到训练集中以提升模型性能:

```bash
# 复制图像
cp {output_dir}/images/* data/GlobalWheat2020/images/train/

# 复制标签
cp {output_dir}/labels/* data/GlobalWheat2020/labels/train/
```

### 方法 2: 创建独立数据集
创建新的数据集配置文件 `wheat_pseudo.yaml`:

```yaml
path: ./data/GlobalWheat2020
train:
  - images/train
  - ../{os.path.basename(output_dir)}/images
val: images/validation
test: images/test

nc: 1
names:
  0: wheat
```

### 方法 3: 迭代训练
1. 使用原始数据训练初始模型
2. 用初始模型生成伪标签
3. 合并伪标签数据重新训练
4. 重复步骤 2-3 进行多轮迭代

## 注意事项
- 伪标签可能存在噪声，建议设置较高的置信度阈值 (≥0.7)
- 可以人工审核部分伪标签以确保质量
- 建议先在小规模数据上测试效果
- 结合原始标注数据一起训练效果更佳

## 建议的训练配置
```python
# 使用伪标签时的训练参数
epochs = 100
batch = 16
imgsz = 1024
conf_threshold = 0.7  # 与生成时保持一致
```
""")
    
    print(f"\n使用说明已保存到: {guide_file}")


def main():
    """主函数"""
    args = parse_args()
    
    # 检查输入
    if not os.path.exists(args.unlabeled_data):
        print(f"错误: 未标注数据目录不存在: {args.unlabeled_data}")
        return
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    # 生成伪标签
    try:
        generate_pseudo_labels(args)
        print("\n✅ 伪标签生成成功!")
    except Exception as e:
        print(f"\n❌ 伪标签生成失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 需要导入 pandas
    import pandas as pd
    main()
