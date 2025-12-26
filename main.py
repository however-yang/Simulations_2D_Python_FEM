import argparse
import time
from typing import Optional, Tuple

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


def _compute_fixed_points(
    rows: int, cols: int, num_fixed: int, active_mask: Optional[np.ndarray]
) -> np.ndarray:
    if active_mask is None:
        return np.stack([np.arange(1, num_fixed + 1), np.ones(num_fixed, dtype=int)], axis=1)
    active_idx = np.argwhere(active_mask)
    if active_idx.size == 0:
        raise ValueError("选择区域为空，无法生成固定点。")
    order = np.lexsort((active_idx[:, 0], active_idx[:, 1]))
    active_idx = active_idx[order]
    chosen = active_idx[: min(num_fixed, active_idx.shape[0])]
    return chosen + 1  # 转为 1-based


def apply_mouse_force(
    SO: SoftObject, inter, drag_k: float, pin_drag: bool = False
) -> Optional[Tuple[int, int]]:
    if inter is None or inter.selected_idx is None:
        return None

    r, c = inter.selected_idx
    if r < 0 or r >= SO.rows or c < 0 or c >= SO.cols:
        return None

    if hasattr(SO, "active") and not SO.active[r, c]:
        return None

    SO.clear_force_ext(r, c)

    if not inter.mouse_down or inter.mouse_pos is None:
        return (r, c)

    if SO.fixed_mode[r, c]:
        return (r, c)

    target = np.array(inter.mouse_pos, dtype=np.float64)
    if pin_drag:
        SO.pos[r, c] = target
        SO.pos_old[r, c] = target
        SO.vel[r, c] = 0.0
        SO.acc[r, c] = 0.0
        SO.force_ext[r, c, :] = 0.0
    else:
        disp = target - SO.pos[r, c]
        SO.force_ext[r, c, :] = disp * drag_k
    return (r, c)


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

    if row >= 50:
        num_fixed = 15
    elif row >= 20:
        num_fixed = 10
    else:
        num_fixed = 5
    pt_fixed_idx = _compute_fixed_points(row, col, num_fixed, active_mask)

    SO = SoftObject(col, row, edge_len, stiffness, damping, pt_fixed_idx, active_mask=active_mask)

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

    t0 = time.time()
    last_force_idx: Optional[Tuple[int, int]] = None
    steps_done = 0

    try:
        for t in range(args.steps):
            inter.process_events()
            if inter.selected_idx is not None and last_force_idx and inter.selected_idx != last_force_idx:
                r_prev, c_prev = last_force_idx
                if 0 <= r_prev < SO.rows and 0 <= c_prev < SO.cols:
                    SO.clear_force_ext(r_prev, c_prev)
                last_force_idx = None
            last_force_idx = apply_mouse_force(SO, inter, args.drag_k, pin_drag=args.pin_drag)

            SO.update_soft_object(mass, ts)

            if inter.quit:
                print("收到按键 'q'，提前结束。")
                steps_done = t + 1
                break

            should_draw = t % args.draw_interval == 0 or inter.mouse_down or inter.cut_pending
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
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--edge-len", type=float, default=0.02)
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--damping", type=float, default=0.5)
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
