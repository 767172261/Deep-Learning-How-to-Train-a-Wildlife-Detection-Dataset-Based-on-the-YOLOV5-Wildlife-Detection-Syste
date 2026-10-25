基于yolov5野生动物检测系统 YOLOV5模型如何训练野生动物检测数据集
@[toc]

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f068efd182434c6b8d67a0d9bd38591f.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c2ce83f1a974e6da25f0e4b3a236352.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5d79177b8394432eaccc7ed46aaefcd6.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/db38049d6ef04c3094b32a9113980186.png)
基于 **YOLOv5** 的野生动物检测系统的完整指南，包含：

- ✅ 数据集说明
- ✅ 环境搭建（CUDA、Anaconda、Python 虚拟环境）
- ✅ YOLOv5 安装与依赖项
- ✅ 数据配置文件 `data.yaml`
- ✅ 训练代码（使用官方 YOLOv5）
- ✅ 推理代码（图片/视频/摄像头）
- ✅ 评估代码
- ✅ 可视化界面（可选 PyQt5 或 Flask）

---

## 🐘 一、数据集信息

| 类别 | ID |
|------|----|
| 水牛 | 0 |
| 大象 | 1 |
| 犀牛 | 2 |
| 斑马 | 3 |

- 格式：YOLO `.txt` 标注文件（归一化坐标）
- 已划分：`train` / `val` / `test`
- 图像格式：`.jpg` 或 `.png`
- 推荐目录结构：

```
wildlife_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
```

---

## 🛠️ 二、环境搭建（CUDA + Anaconda + Python 虚拟环境）

### 1. 安装 CUDA（GPU 支持）

确保 NVIDIA 显卡驱动支持 CUDA。

前往 [NVIDIA CUDA 下载页](https://developer.nvidia.com/cuda-downloads) 安装对应版本。

验证安装：
```bash
nvidia-smi
```

### 2. 安装 Anaconda

下载并安装 [Anaconda](https://www.anaconda.com/products/distribution)

### 3. 创建 Python 虚拟环境并安装依赖

```bash
# 创建虚拟环境
conda create -n yolov5_wildlife python=3.9
conda activate yolov5_wildlife

# 安装 PyTorch（CUDA 11.8 示例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 克隆 YOLOv5 官方仓库
git clone https://github.com/ultralytics/yolov5
cd yolov5

# 安装 YOLOv5 依赖
pip install -r requirements.txt

# 验证安装
python detect.py --source data/images/zidane.jpg
```

---

## 🧩 三、创建 `data.yaml` 文件

在 `wildlife_dataset/` 目录下创建 `data.yaml`：

```yaml
# data.yaml
train: ../wildlife_dataset/images/train
val: ../wildlife_dataset/images/val
test: ../wildlife_dataset/images/test

# 类别数量
nc: 4

# 类别名称（顺序必须与标签 ID 一致）
names:
  - 水牛
  - 大象
  - 犀牛
  - 斑马
```

> ⚠️ 确保路径正确，如果 `data.yaml` 放在 `yolov5/` 目录下，请调整相对路径。

---

## 🚀 四、训练代码（YOLOv5）

### ✅ 使用命令行训练（推荐）

```bash
python train.py \
--img 640 \
--batch 16 \
--epochs 200 \
--data ../wildlife_dataset/data.yaml \
--weights yolov5s.pt \
--device 0 \
--name wildlife_v5s_200e \
--project runs/train \
--patience 50 \
--optimizer AdamW \
--lr0 0.001 \
--augment
```

### ✅ 或使用 Python 脚本训练（`train.py`）

```python
# train_wildlife.py
import torch
from utils.general import check_git_status, check_requirements
from train import run

# 检查依赖
check_requirements()

# 开始训练
run(
    imgsz=640,
    batch_size=16,
    epochs=200,
    data='../wildlife_dataset/data.yaml',
    weights='yolov5s.pt',
    device='0',
    project='runs/train',
    name='wildlife_v5s_200e',
    patience=50,
    optimizer='AdamW',
    lr0=0.001,
    augment=True,
    cache=False
)

print("✅ 野生动物检测模型训练完成！")
```

运行：
```bash
python train_wildlife.py
```

---

## 🔍 五、推理代码（检测新图像/视频）

### 1. 图片推理

```python
# detect_image.py
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
import cv2
import torch

# 加载模型
device = select_device('0')  # 使用 GPU
model = DetectMultiBackend('runs/train/wildlife_v5s_200e/weights/best.pt', device=device)
stride, names = model.stride, model.names

# 加载图像
dataset = LoadImages('test_image.jpg', img_size=640, stride=stride)

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float()  # 半精度
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推理
    pred = model(img)

    # NMS
    pred = non_max_suppression(pred, 0.25, 0.45)

    # 处理结果
    for i, det in enumerate(pred):
        im0 = im0s.copy()
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
                cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('Wildlife Detection', im0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

---

### 2. 视频或摄像头推理

```python
# detect_video.py
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
import cv2
import torch

device = select_device('0')
model = DetectMultiBackend('runs/train/wildlife_v5s_200e/weights/best.pt', device=device)
stride, names = model.stride, model.names

# source = 0  # 摄像头
source = 'wildlife_video.mp4'  # 视频文件

dataset = LoadStreams(source, img_size=640, stride=stride) if source == '0' else LoadImages(source, img_size=640, stride=stride)

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)

    for i, det in enumerate(pred):
        im0 = im0s[i].copy()
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
                cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('Wildlife Detection', im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

---

## 📊 六、评估代码（在测试集上）

```python
# evaluate.py
from utils.general import check_requirements
from val import run

check_requirements()

run(
    data='../wildlife_dataset/data.yaml',
    weights='runs/train/wildlife_v5s_200e/weights/best.pt',
    batch_size=16,
    imgsz=640,
    device='0',
    project='runs/val',
    name='wildlife_eval'
)
```

输出关键指标：
- mAP@0.5
- mAP@0.5:0.95
- Precision, Recall, F1-score

---

## 🎯 七、模型导出（用于部署）

```bash
python export.py \
--weights runs/train/wildlife_v5s_200e/weights/best.pt \
--include onnx torchscript
```

生成：
- `best.onnx` → 用于 C++, Python, Android, iOS
- `best.torchscript` → 用于 PyTorch 推理

---

## 🖼️ 八、可选：构建 GUI 界面（PyQt5）

使用 `PyQt5` 构建一个桌面应用，支持：

- 图像/视频上传
- 实时摄像头检测
- 结果可视化

```bash
pip install PyQt5
```


