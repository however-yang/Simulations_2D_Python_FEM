import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # Matplotlib 可能未安装，非绘图模式下可忽略
    plt = None

from soft_object import SoftObject


class MouseInteractor:
    """Matplotlib 鼠标/键盘交互，保持与 MATLAB 版一致的逻辑。"""

    def __init__(self, ax: Any, soft_obj: SoftObject):
        self.ax = ax
        self.so = soft_obj
        self.mouse_down = False
        self.mouse_pos: Optional[Tuple[float, float]] = None
        self.selected_idx: Optional[Tuple[int, int]] = None
        self.quit = False
        canvas = ax.figure.canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("key_press_event", self._on_key),
        ]

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if getattr(event, "button", None) not in (None, 1):
            return
        self.mouse_down = True
        self.mouse_pos = (event.xdata, event.ydata)
        self.selected_idx = self.so.find_closest_node(event.xdata, event.ydata)

    def _on_release(self, _event):
        self.mouse_down = False

    def _on_motion(self, event):
        if not self.mouse_down or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.mouse_pos = (event.xdata, event.ydata)

    def _on_key(self, event):
        if event.key == "q":
            self.quit = True

    def disconnect(self):
        canvas = self.ax.figure.canvas
        for cid in self._cids:
            canvas.mpl_disconnect(cid)


def build_fracture(SO: SoftObject, row: int, col: int) -> Dict[str, List[int]]:
    """
    依据 MATLAB 逻辑在中上部切一条裂缝（修改 conn_mode）。
    """
    if row >= 50:
        fracture_row = round(row * 0.6)
        fracture_cols = list(range(round(col * 0.4), round(col * 0.6) + 1))
    elif row >= 20:
        fracture_row = round(row * 0.5)
        fracture_cols = list(range(round(col * 0.3), round(col * 0.7) + 1))
    else:
        fracture_row = 10
        fracture_cols = list(range(9, 16))

    # 保证 r 与 r+1 在合法范围
    max_r_for_pair = max(0, row - 2)
    fracture_row = max(0, min(max_r_for_pair, fracture_row))

    # 列坐标裁剪
    fracture_cols = [c for c in fracture_cols if 0 <= c < col]
    if not fracture_cols:
        mid_c = max(0, min(col - 1, col // 2))
        fracture_cols = [mid_c]

    for i, cc in enumerate(fracture_cols):
        r = fracture_row
        c = cc
        if r < 0 or r + 1 >= row:
            continue
        if i == 0:
            SO.set_conn_mode(r, c, 1)
            SO.set_conn_mode(r + 1, c, 7)
        elif i < len(fracture_cols) - 1:
            SO.set_conn_mode(r, c, 2)
            SO.set_conn_mode(r + 1, c, 8)
        else:
            SO.set_conn_mode(r, c, 3)
            SO.set_conn_mode(r + 1, c, 9)
    return {"row": fracture_row, "cols": fracture_cols}


def build_fracture_canvas(SO: SoftObject, fracture_info: Dict[str, List[int]]) -> Optional[np.ndarray]:
    if not fracture_info:
        return None
    r = fracture_info["row"]
    cols = fracture_info["cols"]
    pts: List[np.ndarray] = []
    if r < 0 or r + 1 >= SO.rows:
        return None
    for c in cols:
        if c < 0 or c >= SO.cols:
            continue
        pts.append(SO.pos[r, c])
        pts.append(SO.pos[r + 1, c])
        pts.append(np.array([np.nan, np.nan]))
    if not pts:
        return None
    return np.stack(pts, axis=0)


def apply_mouse_force(
    SO: SoftObject, inter: Optional[MouseInteractor], drag_k: float, pin_drag: bool = False
) -> Optional[Tuple[int, int]]:
    """
    基于当前鼠标状态施加虚拟弹簧外力，返回当前作用的节点索引。
    """
    if inter is None or inter.selected_idx is None:
        return None

    r, c = inter.selected_idx
    if r < 0 or r >= SO.rows or c < 0 or c >= SO.cols:
        return None

    SO.clear_force_ext(r, c)  # 默认清空上一帧力

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


def run_sim(args):
    row = args.rows
    col = args.cols
    edge_len = args.edge_len
    stiffness = args.k
    damping = args.damping
    mass = args.mass
    ts = args.ts

    # 固定点：左边缘若干节点（与 MATLAB 一致）
    if row >= 50:
        num_fixed = 15
    elif row >= 20:
        num_fixed = 10
    else:
        num_fixed = 5
    pt_fixed_idx = np.stack([np.arange(1, num_fixed + 1), np.ones(num_fixed, dtype=int)], axis=1)

    SO = SoftObject(col, row, edge_len, stiffness, damping, pt_fixed_idx)
    fracture_info = build_fracture(SO, row, col)

    fig, ax = None, None
    line_canvas = None
    fixed_scatter = None
    selected_scatter = None
    fracture_line = None
    inter = None

    if args.show:
        if plt is None:
            raise ImportError("需要 matplotlib 才能绘图，请先安装 matplotlib 或去掉 --show 参数。")
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_aspect("equal")
        ax.set_xlim(-0.1, max(row, col) * edge_len + 0.1)
        ax.set_ylim(-0.1, max(row, col) * edge_len + 0.1)
        ax.set_title("Soft Object Simulation (NumPy)")
        fixed_pts = np.argwhere(SO.fixed_mode)
        fixed_xy = SO.pos[fixed_pts[:, 0], fixed_pts[:, 1], :]
        fixed_scatter = ax.scatter(fixed_xy[:, 0], fixed_xy[:, 1], c="k", s=20, label="Fixed")

        canvas = _get_canvas(SO, args.draw_mode, args.draw_skip)
        (line_canvas,) = ax.plot(canvas[:, 0], canvas[:, 1], "r-", linewidth=1)
        frac_canvas = build_fracture_canvas(SO, fracture_info)
        if frac_canvas is None:
            (fracture_line,) = ax.plot([], [], "w-", linewidth=3)
        else:
            (fracture_line,) = ax.plot(frac_canvas[:, 0], frac_canvas[:, 1], "w-", linewidth=3)
        selected_scatter = ax.scatter([], [], c="b", s=40, label="Selected")
        inter = MouseInteractor(ax, SO)

    t0 = time.time()
    last_force_idx: Optional[Tuple[int, int]] = None
    steps_done = 0

    for t in range(args.steps):
        if inter:
            if inter.selected_idx is not None and last_force_idx and inter.selected_idx != last_force_idx:
                r_prev, c_prev = last_force_idx
                if 0 <= r_prev < SO.rows and 0 <= c_prev < SO.cols:
                    SO.clear_force_ext(r_prev, c_prev)
                last_force_idx = None
            last_force_idx = apply_mouse_force(SO, inter, args.drag_k, pin_drag=args.pin_drag)

        SO.update_soft_object(mass, ts)

        if inter and inter.quit:
            print("收到按键 'q'，提前结束。")
            steps_done = t + 1
            break

        should_draw = args.show and (t % args.draw_interval == 0 or (inter and inter.mouse_down))
        if should_draw and ax is not None:
            canvas = _get_canvas(SO, args.draw_mode, args.draw_skip)
            line_canvas.set_data(canvas[:, 0], canvas[:, 1])

            if fracture_line is not None:
                frac_canvas = build_fracture_canvas(SO, fracture_info)
                if frac_canvas is None:
                    fracture_line.set_data([], [])
                else:
                    fracture_line.set_data(frac_canvas[:, 0], frac_canvas[:, 1])

            if fixed_scatter is not None:
                fixed_pts = np.argwhere(SO.fixed_mode)
                fixed_xy = SO.pos[fixed_pts[:, 0], fixed_pts[:, 1], :]
                fixed_scatter.set_offsets(fixed_xy)

            if selected_scatter is not None and inter and inter.selected_idx is not None:
                r_sel, c_sel = inter.selected_idx
                if 0 <= r_sel < SO.rows and 0 <= c_sel < SO.cols:
                    pt_sel = SO.pos[r_sel, c_sel, :]
                    selected_scatter.set_offsets(pt_sel[None, :])
                else:
                    selected_scatter.set_offsets(np.empty((0, 2)))
            elif selected_scatter is not None:
                selected_scatter.set_offsets(np.empty((0, 2)))

            ax.set_title(f"Step {t} | Dragging: {bool(inter and inter.mouse_down)}")
            plt.pause(0.001)

        steps_done = t + 1

    dt = time.time() - t0
    print(f"Simulated {steps_done} steps in {dt:.3f}s (avg {steps_done/dt:.1f} steps/s)")

    if args.show and fig is not None:
        if inter:
            inter.disconnect()
        plt.ioff()
        plt.show()


def _get_canvas(SO: SoftObject, mode: str, draw_skip: int) -> np.ndarray:
    if mode == "full":
        canvas = SO.drawSoftObject()
    elif mode == "contour":
        canvas = SO.drawSoftObjectContour()
    else:
        canvas = SO.drawSoftObjectPt()
    if draw_skip > 1:
        canvas = canvas[::draw_skip]
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--edge-len", type=float, default=0.02)
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=0.01)
    parser.add_argument("--ts", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--draw-interval", type=int, default=15)
    parser.add_argument(
        "--draw-mode",
        choices=["full", "contour", "points"],
        default="full",
        help="full=全部连边，contour=只画边界，points=只画节点",
    )
    parser.add_argument("--draw-skip", type=int, default=1, help="绘制下采样（>1 时跳点/边）")
    parser.add_argument("--drag-k", type=float, default=5.0, help="鼠标拖拽虚拟弹簧系数")
    parser.add_argument("--pin-drag", action="store_true", help="拖拽时将节点直接钉在鼠标位置")
    parser.add_argument("--show", action="store_true", help="开启 Matplotlib 动画（默认仅计算不绘图）")
    args = parser.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
