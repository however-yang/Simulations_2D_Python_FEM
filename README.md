# Python 版软体仿真（NumPy）

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

完全用 NumPy 重写的软体弹簧网格仿真系统，与 MATLAB 版逻辑一一对应。默认依赖只有 `numpy` 和 `matplotlib`，无需 PyTorch/CUDA 即可运行。

## 📋 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [参数说明](#参数说明)
- [实现细节](#实现细节)
- [项目结构](#项目结构)

## ✨ 特性

- 🎯 **纯 NumPy 实现**：无需 PyTorch/CUDA，轻量级依赖
- 🖱️ **交互式仿真**：支持鼠标拖拽节点施加外力
- 🔧 **可配置参数**：丰富的物理参数和渲染选项
- 📊 **多种渲染模式**：完整网格、仅边界、仅节点
- 🔨 **裂缝模拟**：支持预设裂缝，可视化断裂效果
- 🖼️ **纹理贴图**：支持在 2D 网格上贴图并随形变更新
- ✂️ **形状选择**：运行前用鼠标圈选非矩形形状
- ⚡ **高性能计算**：优化的 NumPy 向量化操作

## 📦 安装

### 基础依赖

```bash
pip install numpy matplotlib
```

### 可选依赖（用于 GPU 加速渲染）

```bash
# Vispy 渲染器
pip install vispy

# CUDA-OpenGL 渲染器（需要 CUDA 支持）
pip install pycuda PyOpenGL glfw
```

## 🚀 快速开始

### 基础运行（仅计算，不显示）

```bash
python3 main.py --steps 200
```

### 交互式可视化

```bash
python3 main.py --show --rows 100 --cols 100 --steps 2000
```

**交互操作：**
- 🖱️ **点击**：选择最近的节点
- 🖱️ **按住左键拖拽**：对选中节点施加外力
- ⌨️ **按 `q` 键**：退出程序

## 📖 使用说明

### 基本示例

```bash
# 小网格快速测试
python3 main.py --show --rows 50 --cols 50 --steps 1000

# 大网格高精度仿真
python3 main.py --show --rows 200 --cols 200 --steps 5000 --draw-interval 30

# 仅显示边界轮廓
python3 main.py --show --draw-mode contour --steps 2000

# 仅显示节点
python3 main.py --show --draw-mode points --steps 2000

# 贴图纹理（默认棋盘格）
python3 main.py --show --texture --steps 2000

# 贴图纹理（自定义图片）
python3 main.py --show --texture ./texture.png --steps 2000

# 贴图纹理且隐藏网格线（可选设置）
python3 main.py --show --texture ./texture.png --grid-alpha 0 --steps 2000

# 运行前圈选形状（按 Enter 确认）
python3 main.py --show --select-shape --steps 2000
```

## ⚙️ 参数说明

### 网格参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rows` | 网格行数 | 100 |
| `--cols` | 网格列数 | 100 |
| `--edge-len` | 节点间距 | 0.02 |

### 物理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--k` | 弹簧刚度系数 | 10.0 |
| `--damping` | 阻尼系数 | 0.5 |
| `--mass` | 节点质量 | 0.01 |
| `--ts` | 时间步长 | 0.005 |

### 仿真参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--steps` | 迭代步数 | 2000 |
| `--show` | 开启可视化（不加则仅计算） | False |

### 渲染参数

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--draw-mode` | 渲染模式 | `full` | `full`, `contour`, `points` |
| `--draw-interval` | 绘图更新间隔 | 15 | 正整数 |
| `--draw-skip` | 绘图下采样（>1 减少渲染开销） | 1 | 正整数 |
| `--texture` | 纹理图片路径（不带参数时用棋盘格） | 无 | 路径或省略参数 |
| `--texture-alpha` | 纹理透明度 | 1.0 | 0~1 |
| `--texture-repeat` | 纹理平铺次数 | 1 | 正整数 |
| `--grid-alpha` | 网格线透明度 | 自动 | 0~1 |
| `--select-shape` | 运行前鼠标圈选形状 | False | True/False |

### 交互参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--drag-k` | 鼠标拖拽虚拟弹簧系数 | 5.0 |
| `--pin-drag` | 拖拽时直接将节点钉在鼠标位置 | False |

## 🔧 实现细节

### 数据结构

与 MATLAB 版保持一致的状态变量：
- `pos` / `pos_old`：当前位置和上一帧位置
- `vel` / `acc`：速度和加速度
- `force` / `force_ext`：内部力和外力
- `fixed_mode`：固定节点标记
- `conn_mode`：连接模式（1-9 对应九宫格边界/角落）

### 力计算

- 按 8 方向偏移与 `conn_mode` 过滤
- 计算弹簧形变力并求和
- 支持阻尼和外力叠加

### 裂缝模拟

- `build_fracture` 函数按 MATLAB 规则修改 `conn_mode`
- 绘图时以白线标出断裂位置
- 裂缝位置根据网格大小自适应调整

### 形状选择说明

- 运行时会弹出选择窗口，按住鼠标拖拽圈选区域
- 按 Enter 确认，按 Esc 重置
- 非矩形形状建议使用 `full`/`points` 模式

### 数值积分

使用 Verlet 积分方法：
```
p_new = 2 * p - p_old + dt² * a
```

## 📁 项目结构

```
Simulations_Python/
├── main.py              # 主程序入口
├── soft_object.py       # 软体对象核心类（NumPy 实现）
├── gpu_renderer.py      # 可选：Vispy GPU 渲染器
├── cuda_gl_renderer.py  # 可选：CUDA-OpenGL 渲染器
├── README.md            # 项目说明文档
└── .gitignore           # Git 忽略文件
```

## 📝 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**：本项目是 MATLAB 版本的 Python 移植，保持了相同的物理仿真逻辑和数据结构。
