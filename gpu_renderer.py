"""
Optional GPU-backed rendering using Vispy.
This avoids Matplotlib overhead but still requires copying positions to CPU
before pushing to the GPU draw calls (Vispy expects numpy arrays).
"""
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch

try:
    from vispy import scene
    from vispy.scene import visuals
except Exception as exc:  # pragma: no cover
    raise ImportError("Vispy is required for --gpu-render. Install via `pip install vispy`.") from exc


class VispyInteractor:
    """Mouse/key handler using Vispy events."""

    def __init__(self, canvas: scene.SceneCanvas, so):
        self.canvas = canvas
        self.so = so
        self.mouse_down = False
        self.mouse_pos: Optional[Tuple[float, float]] = None
        self.selected_idx: Optional[Tuple[int, int]] = None
        self.quit = False

        canvas.events.mouse_press.connect(self._on_press)
        canvas.events.mouse_release.connect(self._on_release)
        canvas.events.mouse_move.connect(self._on_move)
        canvas.events.key_press.connect(self._on_key)

    def _on_press(self, event):
        if event.button != 1:
            return
        if event.pos is None:
            return
        x, y = event.pos
        world = self._to_world(x, y)
        self.mouse_down = True
        self.mouse_pos = (world[0], world[1])
        self.selected_idx = self.so.find_closest_node(world[0], world[1])

    def _on_release(self, event):
        if event.button != 1:
            return
        self.mouse_down = False

    def _on_move(self, event):
        if not self.mouse_down or event.pos is None:
            return
        x, y = event.pos
        world = self._to_world(x, y)
        self.mouse_pos = (world[0], world[1])

    def _on_key(self, event):
        if event.key == "Q":
            self.quit = True

    def _to_world(self, x: float, y: float):
        pt = self.canvas.transforms.canvas_transform.map((x, y))
        # map to scene coordinates
        return self.canvas.scene.node.transform.imap((pt[0], pt[1], 0))

    def process_events(self):
        self.canvas.app.process_events()

    def disconnect(self):
        # Vispy handles cleanup on canvas close
        try:
            self.canvas.close()
        except Exception:
            pass


class VispyRenderer:
    """Simple Vispy renderer for lines/points."""

    def __init__(
        self,
        rows: int,
        cols: int,
        edge_len: float,
        draw_mode: str = "full",
        draw_skip: int = 1,
    ):
        self.draw_mode = draw_mode
        self.draw_skip = max(1, draw_skip)

        size = max(rows, cols) * edge_len + 0.1
        self.canvas = scene.SceneCanvas(keys="interactive", show=True, size=(900, 700))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(
            rect=(-0.1, -0.1, size + 0.2, size + 0.2), aspect=1.0
        )

        self.line = visuals.Line(color="red", width=1.0, method="gl")
        self.view.add(self.line)

        self.fixed = visuals.Markers()
        self.fixed.set_data(np.zeros((0, 2), dtype=np.float32), face_color="black", size=6)
        self.view.add(self.fixed)

        self.selected = visuals.Markers()
        self.selected.set_data(np.zeros((0, 2), dtype=np.float32), face_color="blue", size=8)
        self.view.add(self.selected)

        self.fracture = visuals.Line(color="white", width=3.0, method="gl")
        self.view.add(self.fracture)

    def update(
        self,
        so,
        fracture_info: Dict[str, List[int]],
        selected_idx: Optional[Tuple[int, int]],
    ):
        # positions to CPU numpy
        if self.draw_mode == "full":
            canvas = so.drawSoftObject()
            if self.draw_skip > 1:
                canvas = canvas[:: self.draw_skip]
        elif self.draw_mode == "contour":
            canvas = so.drawSoftObjectContour()
        else:
            canvas = so.drawSoftObjectPt()
            if self.draw_skip > 1:
                canvas = canvas[:: self.draw_skip]
        canvas_np = canvas.detach().cpu().numpy()
        self.line.set_data(canvas_np, width=1.0, color="red")

        fixed_pts = so.fixed_mode.nonzero(as_tuple=False).cpu()
        if fixed_pts.numel() > 0:
            fixed_xy = so.pos[fixed_pts[:, 0], fixed_pts[:, 1], :].detach().cpu().numpy()
        else:
            fixed_xy = np.zeros((0, 2), dtype=np.float32)
        self.fixed.set_data(fixed_xy, face_color="black", size=6)

        if selected_idx is not None:
            r_sel, c_sel = selected_idx
            if 0 <= r_sel < so.rows and 0 <= c_sel < so.cols:
                pt_sel = so.pos[r_sel, c_sel, :].detach().cpu().unsqueeze(0).numpy()
            else:
                pt_sel = np.zeros((0, 2), dtype=np.float32)
        else:
            pt_sel = np.zeros((0, 2), dtype=np.float32)
        self.selected.set_data(pt_sel, face_color="blue", size=8)

        frac_canvas = None
        if fracture_info:
            r = fracture_info["row"]
            cols = fracture_info["cols"]
            pts = []
            if 0 <= r + 1 < so.rows:
                for c in cols:
                    if 0 <= c < so.cols:
                        pts.append(so.pos[r, c])
                        pts.append(so.pos[r + 1, c])
                        pts.append(so.pos[r, c] * np.nan)  # NaN separator
            if pts:
                frac_canvas = torch.stack(pts, dim=0)
        if frac_canvas is not None:
            frac_np = frac_canvas.detach().cpu().numpy()
            self.fracture.set_data(frac_np, color="white", width=3.0)
        else:
            self.fracture.set_data(np.zeros((0, 2), dtype=np.float32))

        self.canvas.update()
        self.canvas.app.process_events()
