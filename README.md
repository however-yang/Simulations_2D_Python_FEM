# Python 版软体仿真（GPU‑only / CUDA‑OpenGL 分支）

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本分支专注于 **CUDA‑OpenGL GPU 渲染**。主仿真默认使用 **CuPy 在 GPU 上进行 FEM 计算**（可通过 `--fem-device cpu` 回退到 NumPy），渲染与交互全部通过 GLFW + OpenGL 实现；不再提供 Matplotlib 实时渲染，只保留 Matplotlib Lasso 作为可选的形状预选工具。

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

- ⚡ **GPU‑only 渲染**：OpenGL 纹理网格 + CUDA 互操作
- 🖱️ **交互式仿真**：左键拖拽施加外力（高斯影响范围）
- 🧩 **FEM 核心**：Corotated 三角形单元 + 隐式积分（默认 CuPy GPU 计算）
- 📌 **固定点切换**：运行中按 `f` 键选择/取消固定节点
- ✂️ **交互式切割**：右键移除单元格并重建网格拓扑（真实撕裂）
- 🖼️ **纹理贴图**：纹理随网格形变实时更新
- 💤 **速度睡眠**：小速度自动归零，加速稳定
- 🧭 **形状选择**：运行前圈选非矩形区域（可选）

## 📦 安装

### 基础依赖

```bash
pip install numpy
```

### FEM GPU 计算（默认）

```bash
pip install cupy-cuda11x
```

根据本机 CUDA 版本选择对应的 `cupy-cudaXXX` 包；若只想使用 CPU，可在运行时指定 `--fem-device cpu`。

### 纹理加载（自定义图片时需要）

```bash
pip install pillow
```

### CUDA‑OpenGL 渲染（必需）

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

### 形状预选（可选）

```bash
pip install matplotlib
```

仅在使用 `--select-shape` 时需要 Matplotlib。

## 🚀 快速开始

### CUDA‑OpenGL 纹理渲染（GPU）

```bash
python3 main.py --draw-mode texture --texture ./texture.jpg --draw-interval 1
```

### 点模式（最快）

```bash
python3 main.py --draw-mode points
```

### 轮廓模式

```bash
python3 main.py --draw-mode contour
```

### 运行前圈选形状

```bash
python3 main.py --select-shape --draw-mode texture --texture ./texture.jpg
```

## 🖱️ 交互操作

- **左键按住拖拽**：对选中节点及其周围节点（高斯权重）施加外力
- **按 `f` 键**：切换固定/取消固定节点（基于当前选中的节点）
- **右键点击**：移除所在单元格并重建拓扑（纹理消失 + 真实断裂）
- **按 `q` 键**：退出程序

提示：默认不预设固定点，请在运行中手动选择需要固定的节点。

## 📖 使用说明

```bash
# 小网格快速测试
python3 main.py --rows 50 --cols 50 --steps 1000

# 大网格高精度仿真
python3 main.py --rows 200 --cols 200 --steps 5000 --draw-interval 30

# 仅显示节点
python3 main.py --draw-mode points

# 仅显示轮廓
python3 main.py --draw-mode contour

# 纹理贴图（默认棋盘格）
python3 main.py --draw-mode texture --texture

# 纹理贴图（自定义图片）
python3 main.py --draw-mode texture --texture ./texture.jpg

# 运行前圈选形状
python3 main.py --select-shape --draw-mode texture --texture ./texture.jpg
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
| `--fem-scale` | FEM 刚度缩放（乘到 k） | 1.0 |
| `--poisson` | 泊松比 | 0.3 |
| `--damping` | 阻尼系数 | 0.5 |
| `--sleep-eps` | 速度睡眠阈值（<=0 关闭） | 0.0 |
| `--mass` | 节点质量 | 0.01 |
| `--ts` | 时间步长 | 0.005 |

### 计算参数

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--fem-device` | FEM 计算设备 | `gpu` | `cpu`, `gpu` |

### 仿真参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--steps` | 迭代步数 | 20000 |
| `--select-shape` | 运行前鼠标圈选形状 | False |

### 渲染参数

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--draw-mode` | 渲染模式 | `points` | `points`, `contour`, `texture`, `full` |
| `--draw-interval` | 绘图更新间隔（静止时生效，运动时会强制每步绘制以保证流畅） | 15 | 正整数 |
| `--draw-skip` | 绘图下采样 | 1 | 正整数 |
| `--texture` | 纹理图片路径（仅给参数时使用内置棋盘格） | 无 | 路径或省略参数 |
| `--texture-alpha` | 纹理透明度 | 1.0 | 0~1 |
| `--texture-repeat` | 纹理平铺次数 | 1 | 正整数 |

### 交互参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--drag-k` | 鼠标拖拽虚拟弹簧系数 | 5.0 |
| `--drag-sigma` | 拖拽影响的高斯半径 sigma（世界单位，<=0 退化为单点拖拽） | `2*edge_len` |
| `--drag-radius` | 拖拽影响半径（世界单位，默认 `3*sigma`） | 无 |
| `--pin-drag` | 拖拽时直接将节点钉在鼠标位置 | False |

> 说明：`full` 在 GPU‑only 分支等同于 `points`，保留该选项仅为兼容旧命令。

## 🔧 实现细节

### 总体流程

1. 解析参数，生成规则网格；可选使用 Lasso 生成 `active_mask`。
2. 初始化 `SoftObject`：构建网格状态、活动区域、单元格集合，并生成 FEM 三角形与预计算矩阵。
3. 主循环：
   - 处理输入事件；
   - `apply_mouse_force` 写入外力；
   - `update_soft_object` 执行 FEM 隐式积分与阻尼/睡眠；
   - 渲染更新（CUDA‑OpenGL 或点/轮廓模式）。

### 网格与拓扑

- **规则网格 vs. 节点数组**：物理节点存储在一维数组中，`node_grid_index` 映射到原始网格坐标。
- `cell_nodes` 保存每个四边形单元的四个节点索引；`cell_active` 表示是否有效。
- **切割**：右键移除单元格会触发 `_rebuild_topology`：
  - 根据剩余单元格建立连通分量；
  - 对不同分量**拆点**，让网格在切割处真实断开；
  - 重建三角形索引与 FEM 预计算矩阵，并刷新 GPU 缓冲。

### FEM 与隐式积分

- 使用 **Corotated FEM（三角形单元）** 计算内力。
- 隐式更新形式为 `(M + dt^2 K) x_{t+1} = b`，通过 **共轭梯度(CG)** 求解。
- 速度更新后应用阻尼 `v *= (1 - damping * ts)`（裁剪到 0~1）。
- `--sleep-eps` 速度阈值：低于阈值时速度直接清零以加速稳定。
- 固定节点与非活动节点会被强制保持不动。

### 拖拽外力（高斯分布）

- 左键选中最近节点；拖拽时以该节点为中心计算高斯权重。
- `--drag-sigma` 控制影响范围，`--drag-radius` 设定截断半径。
- 外力形式：`force_ext = k * weight * (target - pos)`。
- `--pin-drag` 可直接把节点钉到鼠标位置。

### CUDA‑OpenGL 渲染

- 使用 VBO 保存顶点坐标，CUDA 将位置直接写入 OpenGL 缓冲。
- 纹理模式下使用网格 UV；`removed_cells` 上传为 **mask 纹理**，片元着色器按单元格 `discard`。
- `CUDAGLInteractor` 处理 GLFW 事件并驱动拖拽/切割/固定操作。

### 形状选择

`--select-shape` 使用 Matplotlib Lasso 生成 `active_mask`：
- 仿真仅在选区内进行；
- 纹理模式仅为选区内**完整单元**生成三角面片。

## 📁 项目结构

```
Simulations_2D_Python_FEM/
├── main.py              # GPU-only 入口：解析参数并直接运行 CUDA 渲染
├── soft_object.py       # 软体网格核心：力计算、切割与遮罩
├── cuda_gl_renderer.py  # CUDA‑OpenGL 渲染器：VBO + shader + mask 纹理
├── gpu_renderer.py      # 旧的 Vispy 实验渲染器（当前未接入 CLI）
├── texture.jpg          # 示例纹理
├── README.md            # 分支说明
└── .gitignore           # Git 忽略配置
```

## 📝 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**：本分支聚焦 GPU 渲染；若需要 Matplotlib 实时渲染版本，请切换到 CPU-only 分支。
