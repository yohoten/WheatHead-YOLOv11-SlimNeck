"""
YOLO11 麦穗检测推理脚本
========================
使用训练好的 YOLO11 模型进行麦穗检测和可视化

使用方法:
    python inference_yolo11.py --model model/yolo11_wheat_exp1/weights/best.pt --source data/GlobalWheat2020/images/test
    
或批量推理并保存结果:
    python inference_yolo11.py --model model/yolo11_wheat_exp1/weights/best.pt \
                               --source data/GlobalWheat2020/images/test \
                               --save-txt --save-conf --project results/inference
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO11 麦穗检测推理脚本')
    
    # 模型配置
    parser.add_argument('--model', type=str, 
                        default='model/yolo11_wheat_exp1/weights/best.pt',
                        help='训练好的模型权重文件路径')
    
    # 数据源
    parser.add_argument('--source', type=str, 
                        default='data/GlobalWheat2020/images/test',
                        help='推理数据源 (图像/目录/视频)')
    
    # 推理配置
    parser.add_argument('--imgsz', '--img-size', type=int, default=1024,
                        help='输入图像尺寸 (默认: 1024)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='NMS IoU 阈值 (默认: 0.45)')
    parser.add_argument('--max-det', type=int, default=300,
                        help='最大检测数量 (默认: 300)')
    parser.add_argument('--device', type=str, default='0',
                        help='推理设备 (默认: 0)')
    parser.add_argument('--half', action='store_true',
                        help='使用 FP16 半精度推理')
    parser.add_argument('--augment', action='store_true',
                        help='启用测试时增强 (TTA)')
    
    # 输出配置
    parser.add_argument('--project', type=str, default='results/inference',
                        help='结果保存目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存检测结果为 txt 文件 (YOLO 格式)')
    parser.add_argument('--save-conf', action='store_true',
                        help='在 txt 文件中保存置信度')
    parser.add_argument('--save-crop', action='store_true',
                        help='保存检测到的目标裁剪图')
    parser.add_argument('--show-labels', action='store_true', default=True,
                        help='显示标签')
    parser.add_argument('--show-conf', action='store_true', default=True,
                        help='显示置信度')
    parser.add_argument('--line-thickness', type=int, default=2,
                        help='边界框线条粗细')
    
    # 可视化
    parser.add_argument('--visualize', action='store_true',
                        help='显示可视化结果')
    parser.add_argument('--num-examples', type=int, default=10,
                        help='可视化示例数量 (默认: 10)')
    
    # 其他
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细信息')
    
    return parser.parse_args()


def load_model(model_path, device='0'):
    """加载 YOLO11 模型"""
    print(f"加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = YOLO(model_path)
    
    # 设置设备
    if device != 'cpu':
        if torch.cuda.is_available():
            print(f"使用 GPU: {torch.cuda.get_device_name(int(device))}")
        else:
            print("警告: CUDA 不可用，将使用 CPU")
            device = 'cpu'
    else:
        print("使用 CPU")
    
    return model


def run_inference(model, source, args):
    """执行推理"""
    print(f"\n开始推理...")
    print(f"数据源: {source}")
    print(f"置信度阈值: {args.conf_thres}")
    print(f"IoU 阈值: {args.iou_thres}")
    print("-" * 80)
    
    # 执行推理
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf_thres,
        iou=args.iou_thres,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        augment=args.augment,
        save=args.save_txt or args.save_conf,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=args.verbose,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        line_width=args.line_thickness,
    )
    
    print(f"\n✅ 推理完成!")
    print(f"结果保存在: {args.project}/{args.name}/")
    
    return results


def visualize_results(results, num_examples=10, save_dir=None):
    """可视化推理结果"""
    print(f"\n可视化 {num_examples} 个示例...")
    
    # 获取前 N 个结果
    example_results = results[:min(num_examples, len(results))]
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(example_results):
        # 获取原始图像
        img = result.orig_img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 绘制边界框
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                # 绘制矩形
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签
                label = f'wheat {conf:.2f}'
                cv2.putText(img_rgb, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示图像
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f'Image {idx + 1}\n{len(boxes) if boxes is not None else 0} wheats detected',
                           fontsize=10)
        axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(example_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {save_path}")
    
    plt.show()


def generate_submission(results, output_file='submission.csv'):
    """生成 Kaggle 提交文件"""
    print(f"\n生成提交文件: {output_file}")
    
    predictions = []
    
    for result in tqdm(results, desc="处理结果"):
        # 获取图像文件名
        img_path = result.path
        img_name = Path(img_path).stem
        
        # 获取检测结果
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # 转换为 COCO 格式: x_min y_min width height
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                width = x2 - x1
                height = y2 - y1
                
                # 格式: image_id confidence x_min y_min width height
                predictions.append({
                    'image_id': img_name,
                    'PredictionString': f'{conf} {x1} {y1} {width} {height}'
                })
        else:
            # 没有检测到任何目标
            predictions.append({
                'image_id': img_name,
                'PredictionString': ''
            })
    
    # 合并同一图像的多个预测
    df = pd.DataFrame(predictions)
    df_grouped = df.groupby('image_id')['PredictionString'].apply(lambda x: ' '.join(x)).reset_index()
    
    # 保存为 CSV
    df_grouped.to_csv(output_file, index=False)
    print(f"提交文件已保存: {output_file}")
    print(f"总图像数: {len(df_grouped)}")
    
    return df_grouped


def calculate_statistics(results):
    """计算检测统计信息"""
    print("\n计算检测统计信息...")
    
    total_images = len(results)
    total_detections = 0
    detections_per_image = []
    
    for result in results:
        boxes = result.boxes
        num_detections = len(boxes) if boxes is not None else 0
        total_detections += num_detections
        detections_per_image.append(num_detections)
    
    avg_detections = total_detections / total_images if total_images > 0 else 0
    
    print(f"总图像数: {total_images}")
    print(f"总检测数: {total_detections}")
    print(f"平均每张图像检测数: {avg_detections:.2f}")
    print(f"最少检测数: {min(detections_per_image)}")
    print(f"最多检测数: {max(detections_per_image)}")
    
    return {
        'total_images': total_images,
        'total_detections': total_detections,
        'avg_detections': avg_detections,
        'min_detections': min(detections_per_image),
        'max_detections': max(detections_per_image),
    }


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("YOLO11 麦穗检测推理")
    print("=" * 80)
    
    try:
        # 1. 加载模型
        model = load_model(args.model, args.device)
        
        # 2. 执行推理
        results = run_inference(model, args.source, args)
        
        # 3. 计算统计信息
        stats = calculate_statistics(results)
        
        # 4. 可视化结果
        if args.visualize:
            save_dir = f"{args.project}/{args.name}"
            visualize_results(results, args.num_examples, save_dir)
        
        # 5. 生成提交文件 (如果是测试集)
        if 'test' in args.source.lower():
            submission_file = f"{args.project}/{args.name}/submission.csv"
            generate_submission(results, submission_file)
        
        print("\n✅ 所有任务完成!")
        
    except Exception as e:
        print(f"\n❌ 推理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
