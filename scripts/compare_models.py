"""
模型对比实验脚本
================
对比 YOLO11、YOLOv5、YOLOv4 和 Faster R-CNN 在麦穗检测任务上的性能

使用方法:
    python compare_models.py
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型对比实验')
    
    parser.add_argument('--data', type=str, 
                        default='data/GlobalWheat2020/wheat.yaml',
                        help='数据集配置文件')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='0',
                        help='设备')
    parser.add_argument('--output', type=str, default='results/comparison',
                        help='结果输出目录')
    
    return parser.parse_args()


class ModelComparator:
    """模型对比器"""
    
    def __init__(self, args):
        self.args = args
        self.results = {}
        self.output_dir = args.output
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 定义要对比的模型
        self.models_config = {
            'YOLO11n': {'model': 'yolo11n.pt', 'imgsz': 640, 'batch': 32},
            'YOLO11s': {'model': 'yolo11s.pt', 'imgsz': 640, 'batch': 16},
            'YOLO11m': {'model': 'yolo11m.pt', 'imgsz': 1024, 'batch': 8},
            'YOLO11l': {'model': 'yolo11l.pt', 'imgsz': 1024, 'batch': 4},
            'YOLO11x': {'model': 'yolo11x.pt', 'imgsz': 1024, 'batch': 2},
            # 'YOLOv5n': {'model': 'yolov5n.pt', 'imgsz': 640, 'batch': 32},
            # 'YOLOv5s': {'model': 'yolov5s.pt', 'imgsz': 640, 'batch': 16},
            # 'YOLOv5m': {'model': 'yolov5m.pt', 'imgsz': 1024, 'batch': 8},
            # 'YOLOv5l': {'model': 'yolov5l.pt', 'imgsz': 1024, 'batch': 4},
            # 'YOLOv5x': {'model': 'yolov5x.pt', 'imgsz': 1024, 'batch': 2},
        }
    
    def train_and_evaluate(self, model_name, config):
        """训练并评估单个模型"""
        print(f"\n{'=' * 80}")
        print(f"训练和评估模型: {model_name}")
        print(f"{'=' * 80}")
        
        model_path = config['model']
        imgsz = config.get('imgsz', self.args.imgsz)
        batch = config.get('batch', self.args.batch)
        
        # 加载模型
        print(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 训练配置
        train_config = {
            'data': self.args.data,
            'epochs': 50,  # 简化实验，使用较少轮数
            'batch': batch,
            'imgsz': imgsz,
            'device': self.args.device,
            'project': self.output_dir,
            'name': f'{model_name}_exp',
            'exist_ok': True,
            'verbose': False,
            'plots': False,
        }
        
        # 训练模型
        print("开始训练...")
        start_time = time.time()
        results = model.train(**train_config)
        training_time = time.time() - start_time
        
        # 获取最佳指标
        metrics = results.metrics
        
        # 推理速度测试
        print("测试推理速度...")
        inference_times = []
        test_images = list(Path('data/GlobalWheat2020/images/test').glob('*.jpg'))[:50]
        
        for img_path in tqdm(test_images, desc="推理速度测试"):
            start = time.time()
            _ = model.predict(str(img_path), imgsz=imgsz, verbose=False)
            end = time.time()
            inference_times.append(end - start)
        
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        # 记录结果
        self.results[model_name] = {
            'model': model_name,
            'mAP50': metrics.get('metrics/mAP50(B)', 0),
            'mAP50_95': metrics.get('metrics/mAP50-95(B)', 0),
            'precision': metrics.get('metrics/precision(B)', 0),
            'recall': metrics.get('metrics/recall(B)', 0),
            'f1_score': metrics.get('metrics/f1(B)', 0),
            'training_time': training_time,
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'imgsz': imgsz,
            'batch': batch,
        }
        
        print(f"✅ {model_name} 完成!")
        print(f"  mAP@0.5: {self.results[model_name]['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {self.results[model_name]['mAP50_95']:.4f}")
        print(f"  FPS: {fps:.2f}")
        
        return self.results[model_name]
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("\n开始模型对比实验...")
        print(f"将测试 {len(self.models_config)} 个模型\n")
        
        for model_name, config in self.models_config.items():
            try:
                self.train_and_evaluate(model_name, config)
            except Exception as e:
                print(f"❌ {model_name} 失败: {str(e)}")
                continue
        
        # 保存结果
        self.save_results()
        
        # 可视化结果
        self.visualize_results()
        
        return self.results
    
    def save_results(self):
        """保存实验结果"""
        results_file = os.path.join(self.output_dir, 'comparison_results.csv')
        
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.to_csv(results_file, index=True)
        
        print(f"\n实验结果已保存到: {results_file}")
        print("\n结果摘要:")
        print(df[['mAP50', 'mAP50_95', 'fps']].to_string())
    
    def visualize_results(self):
        """可视化对比结果"""
        if not self.results:
            print("没有结果可可视化")
            return
        
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        # 1. mAP 对比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # mAP@0.5
        ax1 = axes[0, 0]
        models = df.index.tolist()
        map50 = df['mAP50'].values
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        bars = ax1.barh(models, map50, color=colors)
        ax1.set_xlabel('mAP@0.5', fontsize=12)
        ax1.set_title('mAP@0.5 Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1.0)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, map50)):
            ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
        
        # mAP@0.5:0.95
        ax2 = axes[0, 1]
        map50_95 = df['mAP50_95'].values
        bars = ax2.barh(models, map50_95, color=colors)
        ax2.set_xlabel('mAP@0.5:0.95', fontsize=12)
        ax2.set_title('mAP@0.5:0.95 Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        
        for i, (bar, val) in enumerate(zip(bars, map50_95)):
            ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
        
        # FPS 对比
        ax3 = axes[1, 0]
        fps = df['fps'].values
        bars = ax3.barh(models, fps, color=colors)
        ax3.set_xlabel('FPS (Frames Per Second)', fontsize=12)
        ax3.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
        
        for i, (bar, val) in enumerate(zip(bars, fps)):
            ax3.text(val + 1, i, f'{val:.1f}', va='center', fontsize=9)
        
        # 训练时间对比
        ax4 = axes[1, 1]
        training_time = df['training_time'].values / 60  # 转换为分钟
        bars = ax4.barh(models, training_time, color=colors)
        ax4.set_xlabel('Training Time (minutes)', fontsize=12)
        ax4.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        
        for i, (bar, val) in enumerate(zip(bars, training_time)):
            ax4.text(val + 1, i, f'{val:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
        plt.show()
        
        # 2. 综合性能雷达图
        self.plot_radar_chart(df)
    
    def plot_radar_chart(self, df):
        """绘制雷达图展示综合性能"""
        from math import pi
        
        # 归一化指标
        metrics_to_compare = ['mAP50', 'mAP50_95', 'fps']
        normalized_df = df[metrics_to_compare].copy()
        
        # Min-Max 归一化
        for col in metrics_to_compare:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val - min_val > 0:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 1.0
        
        # 设置角度
        categories = metrics_to_compare
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # 闭合
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(normalized_df)))
        
        for idx, (model_name, row) in enumerate(normalized_df.iterrows()):
            values = row.tolist()
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'radar_chart.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"雷达图已保存到: {save_path}")
        plt.show()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("模型对比实验")
    print("=" * 80)
    
    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: CUDA 不可用，将使用 CPU (速度会很慢)")
    
    # 创建对比器并运行实验
    comparator = ModelComparator(args)
    results = comparator.run_all_experiments()
    
    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
