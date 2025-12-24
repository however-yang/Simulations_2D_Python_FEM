import numpy as np
from typing import Dict, List, Optional, Tuple


class SoftObject:
    """
    纯 NumPy 版软体弹簧网格，直接对应 MATLAB 的数据结构：
    pos/pos_old/vel/acc/force/force_ext/fixed_mode/conn_mode 等。
    """

    def __init__(
        self,
        cols: int,
        rows: int,
        edge_len: float,
        k: float,
        damping: float,
        pt_fixed_idx: Optional[np.ndarray],
        active_mask: Optional[np.ndarray] = None,
    ):
        self.cols = int(cols)
        self.rows = int(rows)
        self.edge_len = float(edge_len)
        self.k_map = np.full((self.rows, self.cols), float(k), dtype=np.float64)
        self.damping_map = np.full((self.rows, self.cols), float(damping), dtype=np.float64)
        self._init_state(pt_fixed_idx)
        self.active = np.ones((self.rows, self.cols), dtype=bool)
        if active_mask is not None:
            self.set_active_mask(active_mask, rebuild=False)
        self._init_conn_masks()

    # ------------------------------------------------------------------ 初始化
    def _init_state(self, pt_fixed_idx: Optional[np.ndarray]):
        r_grid = np.arange(self.rows, dtype=np.float64)
        c_grid = np.arange(self.cols, dtype=np.float64)
        C, R = np.meshgrid(c_grid, r_grid, indexing="xy")  # (rows, cols)

        self.pos_init = np.zeros((self.rows, self.cols, 2), dtype=np.float64)
        self.pos_init[..., 0] = C * self.edge_len
        self.pos_init[..., 1] = R * self.edge_len

        self.pos = self.pos_init.copy()
        self.pos_old = self.pos.copy()
        self.vel = np.zeros_like(self.pos)
        self.acc = np.zeros_like(self.pos)
        self.v = np.zeros_like(self.pos)
        self.force = np.zeros_like(self.pos)
        self.force_ext = np.zeros_like(self.pos)

        self.fixed_mode = np.zeros((self.rows, self.cols), dtype=bool)
        if pt_fixed_idx is not None:
            idx = np.asarray(pt_fixed_idx, dtype=int) - 1  # 转为 0-based
            idx[:, 0] = np.clip(idx[:, 0], 0, self.rows - 1)
            idx[:, 1] = np.clip(idx[:, 1], 0, self.cols - 1)
            self.fixed_mode[idx[:, 0], idx[:, 1]] = True

        # 连接模式（与 MATLAB 保持一致，1~9 分别为九宫格边界/角落）
        conn = np.zeros((self.rows, self.cols), dtype=int)
        for r in range(self.rows):
            for c in range(self.cols):
                if r == self.rows - 1 and c == 0:
                    conn[r, c] = 1
                elif r == self.rows - 1 and c < self.cols - 1:
                    conn[r, c] = 2
                elif r == self.rows - 1 and c == self.cols - 1:
                    conn[r, c] = 3
                elif r > 0 and c == 0:
                    conn[r, c] = 4
                elif r > 0 and c < self.cols - 1:
                    conn[r, c] = 5
                elif r > 0 and c == self.cols - 1:
                    conn[r, c] = 6
                elif r == 0 and c == 0:
                    conn[r, c] = 7
                elif r == 0 and c < self.cols - 1:
                    conn[r, c] = 8
                elif r == 0 and c == self.cols - 1:
                    conn[r, c] = 9
        self.conn_mode = conn

    def _init_conn_masks(self):
        # 与 MATLAB 的 case 分支对应的邻接方向
        self.offset_map: Dict[Tuple[int, int], List[int]] = {
            (0, 1): [1, 2, 4, 5, 7, 8, 10, 11, 12, 13],
            (0, -1): [2, 3, 5, 6, 8, 9, 10, 11, 12, 13],
            (1, 0): [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            (-1, 0): [1, 2, 3, 4, 5, 6, 10, 11, 12, 13],
            (1, 1): [4, 5, 7, 8, 10, 11, 12, 13],
            (1, -1): [5, 6, 8, 9, 10, 11, 12, 13],
            (-1, 1): [1, 2, 4, 5, 10, 12, 13],
            (-1, -1): [2, 3, 5, 6, 10, 11, 12, 13],
        }

        self.offset_masks: Dict[Tuple[int, int], np.ndarray] = {}
        for off, modes in self.offset_map.items():
            mask = np.zeros((self.rows, self.cols), dtype=bool)
            for m in modes:
                mask |= self.conn_mode == m
            self.offset_masks[off] = mask
        if not hasattr(self, "cut_masks") or not self.cut_masks:
            self.cut_masks = {off: np.zeros((self.rows, self.cols), dtype=bool) for off in self.offset_map}
        else:
            for off in self.offset_map:
                if off not in self.cut_masks or self.cut_masks[off].shape != (self.rows, self.cols):
                    self.cut_masks[off] = np.zeros((self.rows, self.cols), dtype=bool)
        self._build_draw_edges()

    def set_active_mask(self, mask: np.ndarray, rebuild: bool = True):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.rows, self.cols):
            raise ValueError("active_mask shape must match (rows, cols).")
        self.active = mask
        self.fixed_mode &= self.active
        if rebuild:
            self._init_conn_masks()

    def _build_draw_edges(self):
        """预计算可绘制的线段（只保存右/上方向，避免重复）。"""
        start_idx: List[int] = []
        end_idx: List[int] = []

        def flat(r: int, c: int) -> int:
            return r * self.cols + c

        ones = np.ones((self.rows, self.cols), dtype=bool)
        for dr, dc in [(0, 1), (1, 0)]:
            neigh_mask = self._shift_with_mask(ones, dr, dc)[1]
            valid = self.offset_masks[(dr, dc)] & neigh_mask
            if hasattr(self, "active"):
                neigh_active, _ = self._shift_with_mask(self.active, dr, dc)
                valid &= self.active & neigh_active
            if hasattr(self, "cut_masks"):
                valid &= ~self.cut_masks[(dr, dc)]
            r_idx, c_idx = np.nonzero(valid)
            for r, c in zip(r_idx.tolist(), c_idx.tolist()):
                start_idx.append(flat(r, c))
                end_idx.append(flat(r + dr, c + dc))

        if start_idx:
            self._draw_edges = (np.array(start_idx, dtype=int), np.array(end_idx, dtype=int))
        else:
            self._draw_edges = None

    # ------------------------------------------------------------------ 更新
    def update_soft_object(self, mass: float, ts: float):
        force_k = self._compute_force_k()
        force_total = force_k - self.damping_map[..., None] * self.vel + self.force_ext
        if hasattr(self, "active"):
            force_total[~self.active] = 0.0
        self.force = force_total

        movable = ~self.fixed_mode
        if hasattr(self, "active"):
            movable &= self.active
        mask3 = movable[..., None]

        acc_new = force_total / mass
        acc_new *= mask3

        p_old = self.pos
        p_new = 2 * self.pos - self.pos_old + (ts * ts) * acc_new
        p_new = np.where(mask3, p_new, p_old)

        vel_new = (p_new - self.pos_old) / (2 * ts)
        vel_new *= mask3

        self.pos_old = p_old
        self.pos = p_new
        self.vel = vel_new
        self.v = self.pos - self.pos_old
        self.acc = acc_new

    # ------------------------------------------------------------------ 力计算
    def _compute_force_k(self) -> np.ndarray:
        force_sum = np.zeros((self.rows, self.cols, 2), dtype=np.float64)

        for (dr, dc), mode_mask in self.offset_masks.items():
            neigh_pos, neigh_mask = self._shift_with_mask(self.pos, dr, dc)
            neigh_init, _ = self._shift_with_mask(self.pos_init, dr, dc)

            valid = mode_mask & neigh_mask
            if hasattr(self, "active"):
                neigh_active, _ = self._shift_with_mask(self.active, dr, dc)
                valid &= self.active & neigh_active
            if hasattr(self, "cut_masks"):
                valid &= ~self.cut_masks[(dr, dc)]
            if not np.any(valid):
                continue

            dir_vec = self.pos - neigh_pos
            dir_norm = np.linalg.norm(dir_vec, axis=2)

            l0_vec = self.pos_init - neigh_init
            l0_norm = np.linalg.norm(l0_vec, axis=2)

            dir_norm_safe = np.where(dir_norm > 1e-12, dir_norm, 1.0)
            stretch = dir_norm - l0_norm
            coeff = -self.k_map * stretch / dir_norm_safe
            force_dir = coeff[..., None] * dir_vec

            force_dir[~valid] = 0.0
            force_sum += force_dir

        return force_sum

    def cut_connection(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        dr = r1 - r0
        dc = c1 - c0
        if (dr, dc) not in self.offset_map:
            return False
        if not (0 <= r0 < self.rows and 0 <= c0 < self.cols):
            return False
        if not (0 <= r1 < self.rows and 0 <= c1 < self.cols):
            return False
        if not self.offset_masks[(dr, dc)][r0, c0]:
            return False
        if hasattr(self, "active"):
            if not (self.active[r0, c0] and self.active[r1, c1]):
                return False
        if self.cut_masks[(dr, dc)][r0, c0]:
            return False
        self.cut_masks[(dr, dc)][r0, c0] = True
        self.cut_masks[(-dr, -dc)][r1, c1] = True
        self._build_draw_edges()
        return True

    def cut_nearest_edge(self, x: float, y: float, max_dist: Optional[float] = None) -> bool:
        r, c = self.find_closest_node(x, y)
        if r < 0 or c < 0:
            return False
        if hasattr(self, "active") and not self.active[r, c]:
            return False

        p = np.array([x, y], dtype=np.float64)
        best = None
        best_dist = np.inf
        best_len = None
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r2 = r + dr
            c2 = c + dc
            if not (0 <= r2 < self.rows and 0 <= c2 < self.cols):
                continue
            if not self.offset_masks[(dr, dc)][r, c]:
                continue
            if self.cut_masks[(dr, dc)][r, c]:
                continue
            if hasattr(self, "active"):
                if not (self.active[r, c] and self.active[r2, c2]):
                    continue
            a = self.pos[r, c]
            b = self.pos[r2, c2]
            ab = b - a
            ab_len2 = float(np.dot(ab, ab))
            if ab_len2 < 1e-12:
                continue
            t = float(np.dot(p - a, ab) / ab_len2)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            dist = float(np.linalg.norm(p - proj))
            if dist < best_dist:
                best_dist = dist
                best = (r2, c2)
                best_len = float(np.sqrt(ab_len2))

        if best is None:
            return False
        limit = max_dist
        if limit is None:
            limit = 0.6 * (best_len if best_len is not None else self.edge_len)
        if best_dist > limit:
            return False
        return self.cut_connection(r, c, best[0], best[1])

    def _shift_with_mask(self, tensor: np.ndarray, dr: int, dc: int) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = tensor.shape[:2]
        shifted = np.zeros_like(tensor)
        mask = np.zeros((rows, cols), dtype=bool)

        r_src_start = max(0, dr)
        r_src_end = min(rows, rows + dr) if dr < 0 else rows
        c_src_start = max(0, dc)
        c_src_end = min(cols, cols + dc) if dc < 0 else cols

        r_dst_start = max(0, -dr)
        r_dst_end = r_dst_start + (r_src_end - r_src_start)
        c_dst_start = max(0, -dc)
        c_dst_end = c_dst_start + (c_src_end - c_src_start)

        shifted[r_dst_start:r_dst_end, c_dst_start:c_dst_end, ...] = tensor[
            r_src_start:r_src_end, c_src_start:c_src_end, ...
        ]
        mask[r_dst_start:r_dst_end, c_dst_start:c_dst_end] = True
        return shifted, mask

    # ------------------------------------------------------------------ 绘图辅助
    def drawSoftObject(self) -> np.ndarray:
        """返回用于 Matplotlib Line2D 的 (N,2) 坐标，使用 NaN 分段。"""
        if not self._draw_edges:
            return np.zeros((0, 2), dtype=np.float64)

        start_idx, end_idx = self._draw_edges
        pos_flat = self.pos.reshape(-1, 2)
        start = pos_flat[start_idx]
        end = pos_flat[end_idx]

        nan_sep = np.full_like(start, np.nan)
        seg = np.stack([start, end, nan_sep], axis=1)
        return seg.reshape(-1, 2)

    def drawSoftObjectContour(self) -> np.ndarray:
        if hasattr(self, "active") and not np.all(self.active):
            return self.drawSoftObject()
        max_points = (self.cols + self.rows) * 2
        canvas = np.zeros((max_points, 2), dtype=np.float64)
        idx = 0
        for c in range(self.cols):
            canvas[idx] = self.pos[0, c]
            idx += 1
        for r in range(self.rows):
            canvas[idx] = self.pos[r, self.cols - 1]
            idx += 1
        for c in range(self.cols - 1, -1, -1):
            canvas[idx] = self.pos[self.rows - 1, c]
            idx += 1
        for r in range(self.rows - 1, -1, -1):
            canvas[idx] = self.pos[r, 0]
            idx += 1
        return canvas[:idx]

    def drawSoftObjectPt(self) -> np.ndarray:
        if hasattr(self, "active"):
            return self.pos[self.active]
        return self.pos.reshape(-1, 2)

    def find_closest_node(self, x: float, y: float) -> Tuple[int, int]:
        target = np.array([x, y], dtype=np.float64)
        diff = self.pos - target
        dist2 = (diff * diff).sum(axis=2)
        if hasattr(self, "active"):
            if not np.any(self.active):
                return -1, -1
            dist2 = np.where(self.active, dist2, np.inf)
        flat_idx = np.argmin(dist2)
        r = int(flat_idx // self.cols)
        c = int(flat_idx % self.cols)
        return r, c

    # ------------------------------------------------------------------ 辅助接口
    def set_force_ext(self, r: int, c: int, fx: float, fy: float):
        self.force_ext[r, c, 0] = fx
        self.force_ext[r, c, 1] = fy

    def clear_force_ext(self, r: int, c: int):
        self.force_ext[r, c, :] = 0.0

    def set_conn_mode(self, r: int, c: int, mode: int):
        self.conn_mode[r, c] = mode
        self._init_conn_masks()
