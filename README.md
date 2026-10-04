# 目标检测-小麦识别说明
## 代码文件说明

- `yolov4-inference.ipynb`：此文件包含了使用YOLOv4模型进行小麦识别的推理代码。该代码可以根据训练好的模型对新的图像进行小麦检测，并输出检测结果。
- `yolov4-training.ipynb`：这是用于训练YOLOv4模型的代码文件。通过使用Global Wheat Head Detection (GWHD)数据集，此代码能够训练一个能够有效识别小麦的YOLOv4模型。
- `yolov4-tiny-training.ipynb`：类似于`yolov4-training.ipynb`，该文件包含了训练YOLOv4-tiny模型的代码。YOLOv4-tiny是一种更小、更快的模型变体，适合资源有限的环境。同样基于GWHD数据集进行训练。

## 对比测试

为了验证模型的性能和准确性，进行了对比测试，包括以下两种方法：

1. `yolov5-pseudo-labeling1.ipynb`：此文件使用YOLOv5模型进行伪标签生成及再训练。参考文献为《Global Wheat Head Detection (GWHD) dataset》。通过伪标签技术，我们能够扩展训练数据集，从而提高模型的泛化能力。
2. `rcnn-inference.ipynb`：此文件实现了基于Region-based Convolutional Neural Networks (RCNN)的小麦检测。参考文献为《Rich feature hierarchies for accurate object detection and semantic segmentation》。RCNN系列模型以其强大的特征提取能力闻名，它们通过构建多尺度特征图来提高检测精度。

# WheatHead-YOLOv11-SlimNeck
