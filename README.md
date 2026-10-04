# WheatHead-YOLOv11-SlimNeck

基于 **Slim-Neck + ECA 注意力**改进的 YOLO11 麦穗检测工程，是毕业设计论文《基于 Slim-Neck 的 YOLOv11 轻量化小麦穗检测》的配套代码与文档仓库。

数据集为 **Global Wheat Head Detection 2020（GWHD）**，单类目标 `wheat`。在 YOLO11n 基线上引入 Slim-Neck 后，参数量 **2.62M → 2.34M（-10.7%）**、计算量**6.6 → 6.2 GFLOPs（-6.1%）**，mAP@0.5 由 0.892 降至 0.870。

| 项目 | 数值 |
|---|---|
| 图像总数 | 6512（train 3655 / val 1476 / test 1381） |
| 标签总数 | 6514（YOLO txt，train 3655 / val 1477 / test 1382） |
| Notebook | 5 个（主实验 1 + 基线对照 4） |
| Python 脚本 | 8 个 |
| 总体积 | 约 17.3 GB（`data/` 占 17 GB，代码与文档约 290 MB） |

---

## 一、要做什么 → 去哪儿找

| 你的目的 | 去这里 |
|---|---|
| 跑通完整实验（训练/评估/推理/出图） | `notebooks/yolov11_wheat_detection.ipynb`（主入口，39 个单元） |
| 看 Slim-Neck + ECA 的网络定义 | `model/yolo11_slimneck.yaml`、`model/custom_modules.py` |
| 命令行训练（本地 / 服务器） | `scripts/train_yolo11.py` |
| 命令行推理与可视化 | `scripts/inference_yolo11.py` |
| 跑论文对比实验（YOLO11/v5/v4/Faster R-CNN） | `scripts/compare_models.py`、`notebooks/baseline_*.ipynb` |
| 生成伪标签做半监督 | `scripts/pseudo_labeling.py` |
| 上云训练（优云智算） | `Dockerfile`、`configs/uyun_task_config.yaml`、`scripts/train_uyun.py`、`scripts/upload_data.py` |
| 看硬件优化参数（RTX 4060 / i9-13900HX） | `scripts/gpu_optimized_config.py`、`scripts/cpu_optimized_config.py` |
| 找论文、开题报告、批注 | `docs/thesis/`、`docs/开题报告.docx`、`docs/批注小结.txt` |
| 找参考文献原文 | `reference/` |

## 二、目录结构

```
WheatHead-YOLOv11-SlimNeck/
├── notebooks/              实验 Notebook
│   ├── yolov11_wheat_detection.ipynb      主实验（环境→数据探索→训练→评估→推理→提交→对比）
│   ├── baseline_yolov5_getting_started.ipynb   GWHD + YOLOv5 入门流程
│   ├── baseline_yolov5_pseudo_labeling.ipynb   YOLOv5 伪标签再训练
│   ├── baseline_yolov4_inference.ipynb         YOLOv4 推理（WBF 融合）
│   └── baseline_rcnn_inference.ipynb           Faster R-CNN 推理
├── scripts/                命令行脚本（8 个 .py）
├── model/                  模型结构定义：yolo11_slimneck.yaml、custom_modules.py
├── configs/                平台配置：uyun_task_config.yaml
├── weights/                预训练权重：yolo11s.pt、yolo11m.pt
├── data/GlobalWheat2020/   数据集（images/ labels/ wheat.yaml）
├── runs/                   训练产物（Ultralytics 自动输出）
├── docs/                   毕设文档：开题报告、要求、批注、数据集种子、thesis/
├── reference/              参考文献 PDF（4 篇）
├── Dockerfile / .dockerignore   优云智算容器配置
├── requirements.txt
└── README.md
```

## 三、数据集

| 划分 | 图像 | 标签 |
|---|---|---|
| train | 3655 | 3655 |
| validation | 1476 | 1477 |
| test | 1381 | 1382 |
| **合计** | **6512** | **6514** |

- 配置：`data/GlobalWheat2020/wheat.yaml`，`nc: 1`，`names: {0: wheat}`
- 标签格式：YOLO 归一化 `class cx cy w h`
- 缓存文件：Ultralytics 在 `images/` 下生成 5131 个 `.npy`、`labels/` 下生成 3 个 `.cache`，属中间产物，可安全删除（下次训练会重建）
- 数据集种子：`docs/Global_Wheat.torrent`

## 四、模型与改进点

`model/yolo11_slimneck.yaml` 在 YOLO11n 基础上做的两处改动：

1. **Slim-Neck**：颈部用 `C3k2` 替换原 C2f/C3 结构，压缩通道与层宽
2. **ECA 注意力**：在 backbone 第 10 层（SPPF 之后）插入 `ECA`（Efficient Channel Attention），卷积核大小由通道数自适应

实测结构指标（`notebooks/yolov11_wheat_detection.ipynb` 单元 34–37 的 `model.info()`）：

| 模型 | 参数量 | GFLOPs |
|---|---|---|
| YOLO11n（基线） | 2,624,080 | 6.6 |
| YOLO11-SlimNeck（本工程） | 2,340,310 | 6.2 |

## 五、实验结果

### 5.1 论文用对比表（来自 Notebook 单元 36）

| Model | Params (M) | GFLOPs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|---|---|
| YOLO11n (Baseline) | 2.62 | 6.6 | 0.892 | 0.497 | 0.897 | 0.811 |
| YOLO11-SlimNeck (Ours) | 2.34 | 6.2 | 0.870 | 0.478 | 0.893 | 0.774 |

参数量 -10.7%，计算量 -6.1%。

### 5.2 落盘训练记录（`runs/`）

| 运行名 | 模型 | imgsz | batch | device | 存档内容 |
|---|---|---|---|---|---|
| `yolo11_wheat_notebook` | yolo11m | 1024 | 8 | cpu | 仅 args.yaml + 训练批次图 |
| `yolo11_wheat_notebook2` | yolo11m | 1024 | 1 | cpu | 仅 args.yaml + 训练批次图 |
| `yolo11_wheat_notebook3` | yolo11s | 640 | 4 | cpu | args.yaml + results.csv + best/last.pt |

`yolo11_wheat_notebook3/results.csv` 只记录了 **4 个 epoch**（0–3），第 3 轮指标为
P 0.863 / R 0.745 / mAP@0.5 0.848 / mAP@0.5:0.95 0.469。

> ⚠️ **注意**：5.1 的对比表与 5.2 的落盘记录**对不上**（对比表的指标没有对应的 results.csv）。
> 论文引用前建议用固定 seed 重跑一次完整训练并保存 `results.csv`，保证数据可复核。

## 六、运行前提

### 6.1 依赖

```bash
pip install -r requirements.txt
# CUDA 11.8 环境另行安装 torch / torchvision / torchaudio
```

### 6.2 工作目录必须是项目根目录（重要）

`notebooks/` 与 `scripts/` 内的路径**全部以项目根为基准**
（`data/GlobalWheat2020/wheat.yaml`、`model/yolo11_slimneck.yaml`、`runs/detect`、`weights/*.pt`）。

- 命令行：在根目录执行 `python scripts/train_yolo11.py ...`
- Notebook：需把 kernel 工作目录设为项目根。VS Code 中设置
  `jupyter.notebookFileRoot` = `${workspaceFolder}`；或 `jupyter lab --notebook-dir .`

### 6.3 预训练权重路径已变更

`scripts/*.py` 的默认 `--model` 仍写作 `yolo11n.pt` / `yolo11s.pt` / `yolo11m.pt`（相对根目录）。
权重现已归入 `weights/`，请显式传入：

```bash
python scripts/train_yolo11.py --model weights/yolo11s.pt --data data/GlobalWheat2020/wheat.yaml
python scripts/train_yolo11.py --model model/yolo11_slimneck.yaml   # 训练改进结构
```

未显式传入时 Ultralytics 会自动联网下载对应权重，离线环境会失败。

### 6.4 缺失路径（需自备或自建）

| 路径 | 状态 |
|---|---|
| `model/yolo11_wheat_exp1/weights/best.pt` | **不存在**。`inference_yolo11.py`、`pseudo_labeling.py` 的默认权重路径，需先训练产出 |
| `weights/yolo11n.pt` | **不存在**。仅 `compare_models.py`、`train_uyun.py` 引用，Ultralytics 会自动下载 |

## 七、已知问题

1. `runs/` 路径异常嵌套为 `runs/detect/model/<name>` —— 训练时 `project` 被设为 `model`，
   与源码目录 `model/` 同名，易混淆。建议后续把 `project` 改为 `runs/detect`。
2. 三次落盘训练全部跑在 **CPU** 上，且 `yolo11_wheat_notebook3` 仅完成 4/50 epoch —— 结果不足以支撑论文结论。
3. `.gitignore` 原为其它项目（PixNamer）的副本，已按本项目重写；数据集与权重默认不入库。
4. `docs/thesis/` 下三份论文 docx 的 md5 各不相同（三个修订版本），未做合并，请自行确认以哪份为准。
5. `data/GlobalWheat2020/labels/output.txt` 来源不明，疑为脚本产物，非标注文件。

## 八、命名约定

- 目录：小写英文、`kebab-case` 或单个单词，职责单一
  （`notebooks` / `scripts` / `model` / `configs` / `weights` / `docs` / `reference`）
- 文件：小写英文 + 下划线；不含空格、括号、全角标点
- 例外：
  - `docs/` 下的中文文件名保留（论文与毕设材料，中文名即检索词）
  - `data/GlobalWheat2020/` 内部结构不动（Ultralytics 按此布局读取）
  - `runs/` 为 Ultralytics 自动输出，不手工改名
  - `weights/yolo11n.pt` 等文件名与 Ultralytics 权重名绑定，不改名

## 九、数字来源

- 文件数、体积：`find` + `du -sh` 统计
- 数据集划分：直接 `ls` 计数 `data/GlobalWheat2020/{images,labels}/*/`
- 训练超参：`runs/detect/model/*/args.yaml`
- 训练指标：`runs/detect/model/yolo11_wheat_notebook3/results.csv`
  （列序为 epoch, time, train/box, train/cls, train/dfl, precision, recall, mAP50, mAP50-95, val/box, val/cls, val/dfl, lr×3）
- 模型结构指标：`notebooks/yolov11_wheat_detection.ipynb` 单元 34–37 的 `model.info()` 输出
