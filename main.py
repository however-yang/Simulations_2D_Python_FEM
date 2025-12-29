import argparse
import time
from typing import Optional

import numpy as np

from soft_object import SoftObject


def _make_checker_texture(size: int = 256, tiles: int = 8) -> np.ndarray:
    size = max(2, int(size))
    tiles = max(1, int(tiles))
    tile = max(1, size // tiles)
    y = np.arange(size)[:, None] // tile
    x = np.arange(size)[None, :] // tile
    mask = (x + y) % 2
    color_a = np.array([0.92, 0.92, 0.92, 1.0], dtype=np.float32)
    color_b = np.array([0.18, 0.18, 0.18, 1.0], dtype=np.float32)
    return np.where(mask[..., None] == 0, color_a, color_b)


def _normalize_texture_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    if np.issubdtype(img.dtype, np.integer):
        max_val = float(np.iinfo(img.dtype).max)
        img = img.astype(np.float32) / max_val
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

    if img.shape[2] == 3:
        alpha = np.ones((*img.shape[:2], 1), dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
    elif img.shape[2] != 4:
        raise ValueError("Texture image must have 1, 3 or 4 channels.")
    return np.clip(img, 0.0, 1.0)


def _load_texture_image(texture_path: str) -> np.ndarray:
    if texture_path == "checker":
        return _make_checker_texture()
    try:
        from PIL import Image
    except Exception as exc:
        raise ImportError("Texture loading requires Pillow in GPU-only branch.") from exc
    img = np.asarray(Image.open(texture_path))
    return _normalize_texture_image(img)


def _select_shape_mask(rows: int, cols: int, edge_len: float) -> Optional[np.ndarray]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.widgets import LassoSelector
    except Exception as exc:
        raise ImportError("形状选择需要 matplotlib，请先安装 matplotlib。") from exc

    x = np.arange(cols, dtype=np.float64) * edge_len
    y = np.arange(rows, dtype=np.float64) * edge_len
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    was_interactive = plt.isinteractive()
    plt.ioff()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, max(rows, cols) * edge_len + 0.1)
    ax.set_ylim(-0.1, max(rows, cols) * edge_len + 0.1)
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c="0.7")
    ax.set_title("Drag and select shape, press Enter to confirm, press Esc to reset")
    selected_scatter = ax.scatter([], [], s=6, c="C0")
    state = {"mask": None}

    def onselect(verts):
        path = Path(verts)
        mask = path.contains_points(pts).reshape(rows, cols)
        state["mask"] = mask
        sel_pts = pts[mask.ravel()]
        selected_scatter.set_offsets(sel_pts)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
        elif event.key == "escape":
            state["mask"] = None
            selected_scatter.set_offsets(np.empty((0, 2)))
            fig.canvas.draw_idle()

    lasso = LassoSelector(ax, onselect)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    lasso.disconnect_events()
    if was_interactive:
        plt.ion()
    return state["mask"]


def apply_mouse_force(
    SO: SoftObject,
    inter,
    drag_k: float,
    pin_drag: bool = False,
    drag_sigma: Optional[float] = None,
    drag_radius: Optional[float] = None,
) -> Optional[int]:
    if inter is None:
        return None

    use_gpu = getattr(SO, "_use_gpu", False) and getattr(SO, "_cp", None) is not None
    if not use_gpu:
        prev_mask = getattr(inter, "_drag_mask", None)
        if prev_mask is not None:
            if prev_mask.shape[0] == SO.force_ext.shape[0]:
                SO.force_ext[prev_mask] = 0.0
            inter._drag_mask = None

        if inter.selected_idx is None:
            return None

        idx = inter.selected_idx
        if idx < 0 or idx >= SO.pos.shape[0]:
            return None

        if hasattr(SO, "active") and not SO.active[idx]:
            return None

        if not inter.mouse_down or inter.mouse_pos is None:
            return idx

        if SO.fixed_mode[idx]:
            return idx

        target = np.array(inter.mouse_pos, dtype=np.float64)
        if pin_drag:
            SO.pos[idx] = target
            SO.pos_old[idx] = target
            SO.vel[idx] = 0.0
            SO.acc[idx] = 0.0
            SO.force_ext[idx, :] = 0.0
        else:
            if drag_sigma is None or drag_sigma <= 0.0:
                disp = target - SO.pos[idx]
                SO.force_ext[idx, :] = disp * drag_k
                mask = np.zeros((SO.pos.shape[0],), dtype=bool)
                mask[idx] = True
                inter._drag_mask = mask
                return idx

            sigma = float(drag_sigma)
            sigma2 = sigma * sigma
            radius = drag_radius
            if radius is None or radius <= 0.0:
                radius = sigma * 3.0
            radius2 = radius * radius

            center = SO.pos[idx]
            diff = SO.pos - center
            dist2 = np.sum(diff * diff, axis=1)
            mask = dist2 <= radius2
            if hasattr(SO, "active"):
                mask &= SO.active
            mask &= ~SO.fixed_mode
            if not np.any(mask):
                return idx

            weights = np.exp(-0.5 * dist2[mask] / sigma2)
            disp = target - SO.pos[mask]
            SO.force_ext[mask] = disp * (drag_k * weights[:, None])
            inter._drag_mask = mask
        return idx

    cp = SO._cp
    cp.cuda.Device().use()
    force_ext_gpu = getattr(SO, "_force_ext_gpu", None)
    if inter.selected_idx is None:
        if force_ext_gpu is not None:
            force_ext_gpu.fill(0.0)
        return None

    idx = inter.selected_idx
    if idx < 0 or idx >= SO.pos.shape[0]:
        return None

    if hasattr(SO, "active") and not SO.active[idx]:
        return None

    if not inter.mouse_down or inter.mouse_pos is None:
        if force_ext_gpu is not None:
            force_ext_gpu.fill(0.0)
        return idx

    if SO.fixed_mode[idx]:
        if force_ext_gpu is not None:
            force_ext_gpu.fill(0.0)
        return idx

    target = np.array(inter.mouse_pos, dtype=np.float64)
    if pin_drag:
        SO.pos[idx] = target
        SO.pos_old[idx] = target
        SO.vel[idx] = 0.0
        SO.acc[idx] = 0.0
        if force_ext_gpu is not None:
            force_ext_gpu.fill(0.0)
        return idx

    if force_ext_gpu is None:
        force_ext_gpu = cp.zeros((SO.pos.shape[0], 2), dtype=cp.float64)
        SO._force_ext_gpu = force_ext_gpu

    if drag_sigma is None or drag_sigma <= 0.0:
        disp = target - SO.pos[idx]
        force_ext_gpu.fill(0.0)
        force_ext_gpu[idx, :] = disp * drag_k
        return idx

    sigma = float(drag_sigma)
    sigma2 = sigma * sigma
    radius = drag_radius
    if radius is None or radius <= 0.0:
        radius = sigma * 3.0
    radius2 = radius * radius

    pos_gpu = cp.asarray(SO.pos, dtype=cp.float64)
    center = pos_gpu[idx]
    diff = pos_gpu - center
    dist2 = cp.sum(diff * diff, axis=1)
    mask = dist2 <= radius2
    if hasattr(SO, "active"):
        mask &= cp.asarray(SO.active)
    mask &= ~cp.asarray(SO.fixed_mode)

    weights = cp.exp(-0.5 * dist2 / sigma2)
    weights = cp.where(mask, weights, 0.0)
    disp = cp.asarray(target, dtype=cp.float64) - pos_gpu
    force_ext_gpu[...] = disp * (drag_k * weights[..., None])
    return idx


def run_sim_cuda(args):
    try:
        from cuda_gl_renderer import CUDAGLInteractor, CUDAGLRenderer
    except Exception as exc:
        raise ImportError("CUDA-OpenGL renderer requires pycuda, PyOpenGL, glfw.") from exc

    row = args.rows
    col = args.cols
    edge_len = args.edge_len
    stiffness = args.k
    damping = args.damping
    mass = args.mass
    ts = args.ts

    active_mask = None
    if args.select_shape:
        active_mask = _select_shape_mask(row, col, edge_len)
        if active_mask is not None and not np.any(active_mask):
            raise ValueError("选择区域为空，无法开始仿真。")

    SO = SoftObject(
        col,
        row,
        edge_len,
        stiffness,
        damping,
        None,
        active_mask=active_mask,
        fem_scale=args.fem_scale,
        poisson=args.poisson,
        fem_device=args.fem_device,
        sleep_eps=args.sleep_eps,
    )

    use_texture = args.texture is not None or args.draw_mode == "texture"
    texture_img = None
    if use_texture:
        tex_path = args.texture if args.texture is not None else "checker"
        texture_img = _load_texture_image(tex_path)
        alpha = float(np.clip(args.texture_alpha, 0.0, 1.0))
        if alpha < 1.0:
            texture_img = texture_img.copy()
            texture_img[..., 3] *= alpha

    draw_mode = args.draw_mode
    if use_texture:
        draw_mode = "texture"
    elif draw_mode == "full":
        draw_mode = "points"

    renderer = CUDAGLRenderer(
        row,
        col,
        edge_len,
        draw_mode=draw_mode,
        draw_skip=args.draw_skip,
        texture_image=texture_img,
        texture_repeat=args.texture_repeat,
        active_mask=active_mask,
    )
    inter = CUDAGLInteractor(renderer, SO, renderer.bounds)

    drag_sigma = args.drag_sigma
    if drag_sigma is None:
        drag_sigma = edge_len * 8.0  # 增大默认影响范围，使多点牵拉效果更明显

    t0 = time.time()
    last_force_idx: Optional[int] = None
    steps_done = 0

    try:
        for t in range(args.steps):
            inter.process_events()
            if inter.selected_idx is not None and last_force_idx is not None and inter.selected_idx != last_force_idx:
                if 0 <= last_force_idx < SO.pos.shape[0]:
                    SO.clear_force_ext(last_force_idx)
                last_force_idx = None
            last_force_idx = apply_mouse_force(
                SO,
                inter,
                args.drag_k,
                pin_drag=args.pin_drag,
                drag_sigma=drag_sigma,
                drag_radius=args.drag_radius,
            )

            SO.update_soft_object(mass, ts)

            if inter.quit:
                print("收到按键 'q'，提前结束。")
                steps_done = t + 1
                break

            moving = False
            if SO.vel.size:
                motion_eps = max(float(args.sleep_eps), 1e-8)
                speed2 = np.sum(SO.vel * SO.vel, axis=1)
                moving = np.any(speed2 > motion_eps * motion_eps)
            should_draw = t % args.draw_interval == 0 or inter.mouse_down or inter.cut_pending or moving
            if should_draw:
                renderer.update(SO, inter.selected_idx)
                inter.cut_pending = False

            steps_done = t + 1
    finally:
        renderer.close()

    dt = time.time() - t0
    print(f"Simulated {steps_done} steps in {dt:.3f}s (avg {steps_done/dt:.1f} steps/s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--cols", type=int, default=200)
    parser.add_argument("--edge-len", type=float, default=0.02)
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--fem-scale", type=float, default=5.0, help="FEM 刚度缩放（乘到 k）")
    parser.add_argument("--poisson", type=float, default=0.45, help="泊松比 (0~0.49)")
    parser.add_argument(
        "--fem-device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="FEM 计算设备（gpu 需要 CuPy）",
    )
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--sleep-eps", type=float, default=0.0, help="速度睡眠阈值（<=0 关闭）")
    parser.add_argument("--mass", type=float, default=0.01)
    parser.add_argument("--ts", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--draw-interval", type=int, default=15)
    parser.add_argument(
        "--draw-mode",
        choices=["full", "contour", "points", "texture"],
        default="points",
        help="points=只画节点，contour=只画边界，texture=纹理网格（CUDA 渲染）",
    )
    parser.add_argument("--draw-skip", type=int, default=1, help="绘制下采样（>1 时跳点/边）")
    parser.add_argument("--drag-k", type=float, default=5.0, help="鼠标拖拽虚拟弹簧系数")
    parser.add_argument(
        "--drag-sigma",
        type=float,
        default=None,
        help="拖拽影响的高斯半径 sigma（世界单位，<=0 退化为单点拖拽）",
    )
    parser.add_argument(
        "--drag-radius",
        type=float,
        default=None,
        help="拖拽影响半径（世界单位，默认 3*sigma）",
    )
    parser.add_argument("--pin-drag", action="store_true", help="拖拽时将节点直接钉在鼠标位置")
    parser.add_argument(
        "--texture",
        nargs="?",
        const="checker",
        default=None,
        help="纹理图片路径；仅给出 --texture 时使用内置棋盘格纹理",
    )
    parser.add_argument("--texture-alpha", type=float, default=1.0, help="纹理透明度 (0~1)")
    parser.add_argument("--texture-repeat", type=int, default=1, help="纹理平铺次数")
    parser.add_argument(
        "--select-shape",
        action="store_true",
        help="运行前使用鼠标选择形状（需 matplotlib）",
    )
    args = parser.parse_args()
    run_sim_cuda(args)


if __name__ == "__main__":
    main()
