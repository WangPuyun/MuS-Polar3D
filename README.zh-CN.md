# MuS-Polar3D

<p align="center">
  <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

[![Paper2](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2512.21513)
[![BaiduPan](https://img.shields.io/badge/Data-BaiduPan-brightgreen)](https://pan.baidu.com/s/1mR-_FX8Ra6PuNPy0fHTREA?pwd=xpu3)


这是一个面向**散射图像法向估计**的深度学习项目。
本仓库集成了多种网络架构（DeepSfP、AttentionU<sup>2</sup>Net、SfPW、TransSfP 等），用于处理和分析散射图像。

<p align="center">
  <img src="./README_img/img1.png" alt="演示图 1" width="310" />
  <img src="./README_img/img2.gif" alt="演示图 2" width="260" />
  <img src="./README_img/img3.png" alt="演示图 3" width="200" />
</p>

## ✨ 特性

* **法向估计**：在 \[-1, 1\] 范围内进行精确的法向预测。
* **多种模型**：包含 DeepSfP、AttentionU<sup>2</sup>Net、SfPW、TransSfP 等模型的实现。
* **训练日志**：自动生成 `training_log_xxx.csv` 文件以记录训练过程。
* **版本管理**：使用 `git_push.sh` 自动生成版本号（v1、v2……）并推送代码，使用 `git_pull.sh` 自动获取并更新至最新代码。

---

## 📦 环境要求

* Python >= 3.9
* PyTorch >= 1.9
* CUDA（推荐用于 GPU 加速）
* 安装依赖：

  ```bash
  pip install -r requirements.txt
  ```

---

## 🚀 使用方法

### 1. 克隆仓库

```bash
git clone https://github.com/WangPuyun/MuS-Polar3D.git
cd MuS-Polar3D
```

### 2. 准备数据集

<p align="center">
  <img src="./README_img/Dataset.png" alt="数据集结构" width="910" />
</p>

百度网盘中的数据集结构如下：

```
MuS-Polar3D
│
├───MuS-Polar3D_Mat/        # MATLAB .mat 文件（847 个）
│   ├─ Angel1_back_0.mat
│   ├─ Angel1_back_1.mat
│   ├─ ...
│
├───MuS-Polar3D_Pol/        # 偏振图像（847 个文件夹）
│   ├───Angel1_back_0/
│   │   ├─ 0.png
│   │   ├─ 45.png
│   │   ├─ 90.png
│   │   ├─ 135.png
│   │   ├─ mask.png
│   │   └─ Normal_gt.png
│   ├───Angel1_back_1/
│   ├─ ...
│
├───MuS-Polar3D_Raw/        # 原始采集数据（121 个文件夹）
│   ├───Angel1_left/
│   │   ├─ Angel1_left_0.bmp
│   │   ├─ Angel1_left_1.bmp
│   │   ├─ ...
│   │   ├─ Angel2_left.mlp
│   │   ├─ mask.png
│   │   ├─ normal.exr
│   │   └─ Normal_gt.png
│   ├───Angel1_right/
│   ├───Angel2_back/
│   ├─ ...
│
├───MuS-Polar3D_3D/         # 3D 物体模型（.obj，共 42 个）
    ├─ Angel1.obj
    ├─ Angel2.obj
    ├─ ...
```

#### 说明

- 📊 **MuS-Polar3D_Mat**：处理后的 `.mat` 文件（**847 个**），包含偏振强度图像（0°、45°、90°、135°）、物体掩码、真实法向、偏振参数及相关物理分量。
- 🖼️ **MuS-Polar3D_Pol**：偏振图像数据集，每个文件夹包含 4 个角度（**0°、45°、90°、135°**）的图像、分割掩码和真实法向（**847 个文件夹**）。
- 📂 **MuS-Polar3D_Raw**：原始采集数据，包括 `.bmp` 序列、`.mlp` 工程文件、分割掩码、EXR 法向图和真实法向（**121 个文件夹**）。
- 🧩 **MuS-Polar3D_3D**：`.obj` 格式的 3D 物体模型（**42 个**），提供与偏振图像对应的几何参考。

> 运行代码前，需要准备实验所用的基线数据集。请按以下步骤操作：
>
> 1. 从提供的[百度网盘链接](https://pan.baidu.com/s/1mR-_FX8Ra6PuNPy0fHTREA?pwd=xpu3)下载全部 `.mat` 数据文件。
> 2. 将**所有 `.mat` 文件**放入 `Underwater Dataset/Baseline_Data/` 目录。

### 3. 训练模型

```bash
python train.py
```

### 4. 评估与可视化

你可以直接使用我们的[预训练模型](https://pan.baidu.com/s/1viZRWyxiZrgvYLsuaVpEJQ?pwd=i3db)进行评估和可视化，无需重新训练。

可以使用 `Angle_error_map.py` 脚本评估表面法向图，并可视化对应的角度误差。该脚本主要完成两项任务：

1. 根据输入数据生成**表面法向图**。
2. 生成**逐像素角度误差彩色图**，直观展示真实法向与预测法向之间的角度误差。

```bash
python Angle_error_map.py
```

<p align="center">
  <img src="./README_img/Angle_error_map.png" alt="角度误差图" width="910" />
</p>

### 5. 二次开发

如果你计划使用自己的数据在本仓库基础上继续开发：

- 将图像数据保存为 MATLAB `.mat` 文件。
- 在 `Dataset.py` 中读取 `.mat` 文件内的相应变量（根据你的 `.mat` 结构调整变量名或键名）。
- 在 `config.py` 中修改或切换要训练的基线网络模型配置，然后照常运行 `train.py`。

---

## 🙏 致谢

本项目部分使用了以下开源项目的实现：

- [Deep Shape from Polarization](https://github.com/UCLA-VMG/DeepSfP.git)
- [Shape from Polarization for Complex Scenes in the Wild](https://github.com/ChenyangLEI/sfp-wild.git)
- [Transparent Shape from a Single View Polarization Image](https://github.com/shaomq2187/TransSfP.git)

衷心感谢上述项目的作者开源代码，为本研究提供了重要支持。

---

## 🤝 引用

如果本项目对你的研究有所帮助，请考虑引用：

```bibtex
@article{wang2025mus,
  title={MuS-Polar3D: A Benchmark Dataset for Computational Polarimetric 3D Imaging under Multi-Scattering Conditions},
  author={Wang, Puyun and Yu, Kaimin and He, Huayang and Wu, Xianyu},
  journal={arXiv preprint arXiv:2512.21513},
  year={2025}
}
```
