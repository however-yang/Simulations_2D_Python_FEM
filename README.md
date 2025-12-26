# Python 版软体仿真（NumPy）

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于弹簧‑质点网格的 2D 软体仿真。核心计算使用 NumPy，默认只依赖 `numpy` 和 `matplotlib`。支持纹理贴图、鼠标交互、形状预选，以及 CUDA‑OpenGL 的 GPU 纹理渲染与切割遮罩。

## 📋 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [交互操作](#交互操作)
- [使用说明](#使用说明)
- [参数说明](#参数说明)
- [实现细节](#实现细节)
- [项目结构](#项目结构)

## ✨ 特性

- 🎯 **纯 NumPy 计算**：主仿真无需 PyTorch/CUDA
- 🖱️ **交互式仿真**：左键拖拽施加外力
- ✂️ **交互式切割**：右键移除单元格并触发弹簧断裂
- 🖼️ **纹理贴图**：纹理随网格形变实时更新
- 🧭 **形状选择**：运行前用鼠标圈选非矩形区域
- ⚡ **CUDA‑OpenGL 渲染**：GPU 纹理网格渲染 + 片元遮罩裁剪

## 📦 安装

### 基础依赖

```bash
pip install numpy matplotlib
```

纹理加载若不安装 `matplotlib`，可使用 Pillow：

```bash
pip install pillow
```

### 可选依赖（CUDA‑OpenGL 渲染）

```bash
pip install pycuda PyOpenGL glfw
```

CUDA‑OpenGL 互操作需要 PyCUDA 启用 GL 扩展，验证方式：

```bash
python3 - <<'PY'
import pycuda._driver as drv
print(drv.have_gl_ext())
PY
```

若输出为 `False`，需从源码重新编译 PyCUDA 并开启 GL 支持（确保 `CUDA_ENABLE_GL = True`）。

## 🚀 快速开始

### 仅计算（不显示）

```bash
python3 main.py --steps 200
```

### Matplotlib 可视化

```bash
python3 main.py --show --rows 100 --cols 100 --steps 2000
```

### 纹理贴图（内置棋盘格）

```bash
python3 main.py --show --texture --steps 2000
```

### 纹理贴图（自定义图片）

```bash
python3 main.py --show --texture ./texture.jpg --grid-alpha 0 --steps 2000
```

### CUDA‑OpenGL 纹理渲染（GPU）

```bash
python3 main.py --renderer cuda --draw-mode texture --texture ./texture.jpg --draw-interval 1
```

## 🖱️ 交互操作

- **左键按住拖拽**：对选中节点施加外力
- **右键点击**：移除所在单元格（纹理消失 + 弹簧断裂）
- **按 `q` 键**：退出程序

## 📖 使用说明

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
python3 main.py --show --texture ./texture.jpg --steps 2000

# 运行前圈选形状（按 Enter 确认）
python3 main.py --show --select-shape --steps 2000

# CUDA‑OpenGL 纹理渲染（GPU）
python3 main.py --renderer cuda --draw-mode texture --texture ./texture.jpg --steps 2000
```

## ⚙️ 参数说明

### 网格参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rows` | 网格行数 | 20 |
| `--cols` | 网格列数 | 20 |
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
| `--steps` | 迭代步数 | 20000 |
| `--show` | 开启 Matplotlib 可视化 | False |
| `--select-shape` | 运行前鼠标圈选形状 | False |

### 渲染参数

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--draw-mode` | 渲染模式 | `full` | `full`, `contour`, `points`, `texture` |
| `--draw-interval` | 绘图更新间隔 | 15 | 正整数 |
| `--draw-skip` | 绘图下采样 | 1 | 正整数 |
| `--texture` | 纹理图片路径（仅给参数时使用内置棋盘格） | 无 | 路径或省略参数 |
| `--texture-alpha` | 纹理透明度 | 1.0 | 0~1 |
| `--texture-repeat` | 纹理平铺次数 | 1 | 正整数 |
| `--grid-alpha` | 网格线透明度 | 自动 | 0~1 |
| `--renderer` | 渲染后端 | `mpl` | `mpl`, `cuda` |

### 交互参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--drag-k` | 鼠标拖拽虚拟弹簧系数 | 5.0 |
| `--pin-drag` | 拖拽时直接将节点钉在鼠标位置 | False |

## 🔧 实现细节

### 数据结构

- `pos` / `pos_old`：当前位置与上一帧位置
- `vel` / `acc`：速度与加速度
- `force` / `force_ext`：内部力与外力
- `fixed_mode`：固定节点标记
- `conn_mode`：连接模式（九宫格边界/角落）

### 力计算

按 8 方向邻接计算弹簧力，结合阻尼与外力后更新 Verlet 积分。

### 切割逻辑（单元格移除）

- 右键点击会将最近的**单元格**加入 `removed_cells`。
- **水平/垂直弹簧**：仅当两侧相邻单元格都被移除时断裂；边缘弹簧仅需一侧单元格移除即断裂。
- **对角弹簧**：所在单元格被移除即断裂。
- `cut_masks` 用于屏蔽力计算与绘制。

### 纹理渲染

- Matplotlib：用 `pcolormesh` 绘制单元格纹理，依据 `get_cut_cell_mask()` 将被移除的单元格透明化。
- CUDA‑OpenGL：上传 `removed_cells` 为 mask 纹理，片元着色器中基于单元格索引 `discard`，从 GPU 侧实时裁剪纹理区域。

### 形状选择

运行前弹出 Lasso 选择窗口，生成 `active_mask`：
- 仿真只在选区内计算
- 固定点优先选在选区最左侧节点
- 纹理与网格仅显示有效区域

## 📁 项目结构

```
Simulations_2D_Python/
├── main.py              # 主入口：参数解析、仿真循环、Matplotlib 渲染
├── soft_object.py       # 软体网格核心：力计算、切割与遮罩
├── cuda_gl_renderer.py  # CUDA‑OpenGL 渲染器：VBO + shader + mask 纹理
├── gpu_renderer.py      # 实验性 Vispy 渲染器（当前未接入 CLI）
├── texture.jpg          # 示例纹理
├── README.md            # 项目说明文档
└── .gitignore           # Git 忽略配置
```

## 📝 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**：本项目为 MATLAB 版本的 Python 移植，保持了相同的物理仿真逻辑与数据结构。
