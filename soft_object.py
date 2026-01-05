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
        fem_scale: float = 1.0,
        poisson: float = 0.3,
        fem_device: str = "cpu",
        sleep_eps: float = 0.0,
        pin_sigma: Optional[float] = None,
        pin_radius: Optional[float] = None,
    ):
        self.cols = int(cols)
        self.rows = int(rows)
        self.edge_len = float(edge_len)
        self.k_map = np.full((self.rows, self.cols), float(k), dtype=np.float64)
        self.damping_map = np.full((self.rows, self.cols), float(damping), dtype=np.float64)
        self.fem_scale = max(0.0, float(fem_scale))
        self.poisson = float(np.clip(poisson, 0.0, 0.49))
        self.sleep_eps = max(0.0, float(sleep_eps))
        if pin_sigma is None:
            pin_sigma = self.edge_len * 4.0
        self.pin_sigma = max(0.0, float(pin_sigma))
        if pin_radius is None:
            pin_radius = self.pin_sigma * 3.0
        self.pin_radius = max(0.0, float(pin_radius))
        self.pin_strength = max(1.0, self.fem_scale)
        self._use_gpu = False
        self._cp = None
        self._tri_gpu = None
        self._force_ext_gpu = None
        self._mesh_version = 0
        self._pending_rebuild = False
        self._node_grid_index = None
        self._init_state(pt_fixed_idx)
        self.active_grid = np.ones((self.rows, self.cols), dtype=bool)
        if active_mask is not None:
            self.set_active_mask(active_mask, rebuild=False)
        self._init_removed_cells()
        self._rebuild_topology(preserve_state=False)
        self._init_fem_backend(fem_device)
        self._update_pin_from_fixed_grid()

    # ------------------------------------------------------------------ 初始化
    def _init_state(self, pt_fixed_idx: Optional[np.ndarray]):
        r_grid = np.arange(self.rows, dtype=np.float64)
        c_grid = np.arange(self.cols, dtype=np.float64)
        C, R = np.meshgrid(c_grid, r_grid, indexing="xy")  # (rows, cols)

        self.pos_init_grid = np.zeros((self.rows, self.cols, 2), dtype=np.float64)
        self.pos_init_grid[..., 0] = C * self.edge_len
        self.pos_init_grid[..., 1] = R * self.edge_len

        self.pin_weight_grid = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.pin_target_grid = self.pos_init_grid.copy()

        self.fixed_grid = np.zeros((self.rows, self.cols), dtype=bool)
        if pt_fixed_idx is not None:
            idx = np.asarray(pt_fixed_idx, dtype=int) - 1  # 转为 0-based
            idx[:, 0] = np.clip(idx[:, 0], 0, self.rows - 1)
            idx[:, 1] = np.clip(idx[:, 1], 0, self.cols - 1)
            self.fixed_grid[idx[:, 0], idx[:, 1]] = True

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

    def _init_removed_cells(self):
        cell_shape = (max(self.rows - 1, 0), max(self.cols - 1, 0))
        self.removed_cells = np.zeros(cell_shape, dtype=bool)
        self.removed_cells_version = 0

    def _rebuild_topology(self, preserve_state: bool = True):
        cell_rows = max(self.rows - 1, 0)
        cell_cols = max(self.cols - 1, 0)
        cell_shape = (cell_rows, cell_cols)

        cell_active = np.zeros(cell_shape, dtype=bool)
        if cell_rows > 0 and cell_cols > 0:
            cell_active[:, :] = True
            if self.active_grid is not None:
                active = self.active_grid
                cell_active &= (
                    active[:-1, :-1]
                    & active[1:, :-1]
                    & active[:-1, 1:]
                    & active[1:, 1:]
                )
            if getattr(self, "removed_cells", None) is not None:
                if self.removed_cells.shape != cell_shape:
                    self.removed_cells = np.zeros(cell_shape, dtype=bool)
                    self.removed_cells_version = 0
                cell_active &= ~self.removed_cells
        self.cell_active = cell_active

        comp = -np.ones(cell_shape, dtype=int)
        comp_id = 0
        for r in range(cell_rows):
            for c in range(cell_cols):
                if not cell_active[r, c] or comp[r, c] >= 0:
                    continue
                stack = [(r, c)]
                comp[r, c] = comp_id
                while stack:
                    rr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr = rr + dr
                        nc = cc + dc
                        if nr < 0 or nr >= cell_rows or nc < 0 or nc >= cell_cols:
                            continue
                        if not cell_active[nr, nc] or comp[nr, nc] >= 0:
                            continue
                        comp[nr, nc] = comp_id
                        stack.append((nr, nc))
                comp_id += 1

        node_map: Dict[Tuple[int, int], int] = {}
        node_grid_index: List[int] = []
        cell_nodes = np.full((cell_rows, cell_cols, 4), -1, dtype=int)

        def grid_idx(rr: int, cc: int) -> int:
            return rr * self.cols + cc

        for r in range(cell_rows):
            for c in range(cell_cols):
                if not cell_active[r, c]:
                    continue
                cid = comp[r, c]
                corners = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                node_ids = []
                for rr, cc in corners:
                    g_idx = grid_idx(rr, cc)
                    key = (cid, g_idx)
                    if key not in node_map:
                        node_map[key] = len(node_grid_index)
                        node_grid_index.append(g_idx)
                    node_ids.append(node_map[key])
                cell_nodes[r, c] = node_ids

        if cell_rows == 0 or cell_cols == 0:
            node_grid_index = list(range(self.rows * self.cols))
            cell_nodes = np.full((cell_rows, cell_cols, 4), -1, dtype=int)

        node_grid_index_arr = np.asarray(node_grid_index, dtype=int)
        node_count = int(node_grid_index_arr.size)
        grid_count = self.rows * self.cols

        pos_ref = self.pos_init_grid.reshape(-1, 2)
        vel_ref = np.zeros_like(pos_ref)
        if preserve_state and getattr(self, "node_grid_index", None) is not None and getattr(self, "pos", None) is not None:
            old_grid = self.node_grid_index
            if old_grid.size > 0:
                counts = np.bincount(old_grid, minlength=grid_count).astype(np.float64)
                pos_sum = np.zeros((grid_count, 2), dtype=np.float64)
                np.add.at(pos_sum, old_grid, self.pos)
                vel_sum = np.zeros((grid_count, 2), dtype=np.float64)
                np.add.at(vel_sum, old_grid, self.vel)
                valid = counts > 0
                pos_ref = self.pos_init_grid.reshape(-1, 2).copy()
                pos_ref[valid] = pos_sum[valid] / counts[valid, None]
                vel_ref = np.zeros_like(pos_ref)
                vel_ref[valid] = vel_sum[valid] / counts[valid, None]

        self.node_grid_index = node_grid_index_arr
        if node_count > 0:
            self.pos_init = self.pos_init_grid.reshape(-1, 2)[self.node_grid_index].copy()
            self.pos = pos_ref[self.node_grid_index].copy()
            self.pos_old = self.pos.copy()
            self.vel = vel_ref[self.node_grid_index].copy()
            self.acc = np.zeros_like(self.pos)
            self.v = np.zeros_like(self.pos)
            self.force = np.zeros_like(self.pos)
            self.force_ext = np.zeros_like(self.pos)
            fixed_flat = self.fixed_grid.reshape(-1)
            self.fixed_mode = fixed_flat[self.node_grid_index].copy()
            self.active = np.ones(node_count, dtype=bool)
            k_flat = self.k_map.reshape(-1)
            self.k_nodes = k_flat[self.node_grid_index].copy()
            damp_flat = self.damping_map.reshape(-1)
            self.damping_nodes = damp_flat[self.node_grid_index].copy()
        else:
            self.pos_init = np.zeros((0, 2), dtype=np.float64)
            self.pos = self.pos_init.copy()
            self.pos_old = self.pos.copy()
            self.vel = np.zeros_like(self.pos)
            self.acc = np.zeros_like(self.pos)
            self.v = np.zeros_like(self.pos)
            self.force = np.zeros_like(self.pos)
            self.force_ext = np.zeros_like(self.pos)
            self.fixed_mode = np.zeros((0,), dtype=bool)
            self.active = np.zeros((0,), dtype=bool)
            self.k_nodes = np.zeros((0,), dtype=np.float64)
            self.damping_nodes = np.zeros((0,), dtype=np.float64)

        self._sync_pin_nodes()
        self.cell_nodes = cell_nodes
        self._init_fem_mesh()
        if self._use_gpu:
            if self._cp is not None:
                self._cp.cuda.Device().use()
            self._tri_gpu = self._build_fem_gpu_buffers()
            if self._cp is not None:
                self._force_ext_gpu = self._cp.zeros(self.force_ext.shape, dtype=self._cp.float64)
        self._mesh_version = int(getattr(self, "_mesh_version", 0)) + 1

    def _ensure_pin_grids(self):
        if not hasattr(self, "pin_weight_grid") or self.pin_weight_grid.shape != (self.rows, self.cols):
            self.pin_weight_grid = np.zeros((self.rows, self.cols), dtype=np.float64)
        if not hasattr(self, "pin_target_grid") or self.pin_target_grid.shape != (self.rows, self.cols, 2):
            self.pin_target_grid = self.pos_init_grid.copy()

    def _grid_positions_from_nodes(self) -> np.ndarray:
        grid_count = self.rows * self.cols
        pos_grid = self.pos_init_grid.reshape(-1, 2).copy()
        if self.node_grid_index is None or self.node_grid_index.size == 0:
            return pos_grid.reshape(self.rows, self.cols, 2)
        counts = np.bincount(self.node_grid_index, minlength=grid_count).astype(np.float64)
        pos_sum = np.zeros((grid_count, 2), dtype=np.float64)
        np.add.at(pos_sum, self.node_grid_index, self.pos)
        valid = counts > 0
        pos_grid[valid] = pos_sum[valid] / counts[valid, None]
        return pos_grid.reshape(self.rows, self.cols, 2)

    def _compute_pin_weight_grid(self) -> np.ndarray:
        if self.pin_sigma <= 0.0:
            return np.zeros((self.rows, self.cols), dtype=np.float64)
        centers = np.argwhere(self.fixed_grid)
        if centers.size == 0:
            return np.zeros((self.rows, self.cols), dtype=np.float64)
        sigma2 = self.pin_sigma * self.pin_sigma
        radius2 = self.pin_radius * self.pin_radius if self.pin_radius > 0.0 else None
        grid_pos = self.pos_init_grid
        weights = np.zeros((self.rows, self.cols), dtype=np.float64)
        for r, c in centers:
            center = grid_pos[r, c]
            diff = grid_pos - center
            dist2 = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1]
            w = np.exp(-0.5 * dist2 / sigma2)
            if radius2 is not None:
                w = np.where(dist2 <= radius2, w, 0.0)
            weights = np.maximum(weights, w)
        if self.active_grid is not None:
            weights *= self.active_grid
        return weights

    def _sync_pin_nodes(self):
        self._ensure_pin_grids()
        if self.node_grid_index is None or self.node_grid_index.size == 0:
            self.pin_weight = np.zeros((0,), dtype=np.float64)
            self.pin_target = np.zeros((0, 2), dtype=np.float64)
            return
        pin_weight_flat = self.pin_weight_grid.reshape(-1)
        self.pin_weight = pin_weight_flat[self.node_grid_index].copy()
        pin_target_flat = self.pin_target_grid.reshape(-1, 2)
        self.pin_target = pin_target_flat[self.node_grid_index].copy()

    def _update_pin_from_fixed_grid(self):
        self._ensure_pin_grids()
        prev_weight = self.pin_weight_grid.copy()
        self.pin_weight_grid = self._compute_pin_weight_grid()
        increased = self.pin_weight_grid > prev_weight
        if np.any(increased):
            pos_grid = self._grid_positions_from_nodes()
            self.pin_target_grid[increased] = pos_grid[increased]
        self._sync_pin_nodes()

    def _init_fem_mesh(self):
        if self.cell_nodes is None or self.rows < 2 or self.cols < 2:
            self._tri_indices = None
            self._tri_inv_dm = None
            self._tri_ke = None
            self._tri_ke_x0 = None
            self._tri_x0 = None
            return

        tris: List[List[int]] = []
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                if not self.cell_active[r, c]:
                    continue
                v00, v10, v01, v11 = self.cell_nodes[r, c]
                if v00 < 0 or v10 < 0 or v01 < 0 or v11 < 0:
                    continue
                tris.append([v00, v10, v11])
                tris.append([v00, v11, v01])
        if tris:
            self._tri_indices = np.asarray(tris, dtype=int)
        else:
            self._tri_indices = None
        self._build_fem_matrices()

    def _build_fem_matrices(self):
        if self._tri_indices is None or self._tri_indices.size == 0:
            self._tri_inv_dm = None
            self._tri_ke = None
            self._tri_ke_x0 = None
            self._tri_x0 = None
            return

        idx0 = self._tri_indices[:, 0]
        idx1 = self._tri_indices[:, 1]
        idx2 = self._tri_indices[:, 2]

        if self.pos_init is None or self.pos_init.size == 0:
            self._tri_inv_dm = None
            self._tri_ke = None
            self._tri_ke_x0 = None
            self._tri_x0 = None
            return

        x0_all = self.pos_init
        x0 = np.stack([x0_all[idx0], x0_all[idx1], x0_all[idx2]], axis=1)
        self._tri_x0 = x0

        dm = np.stack([x0[:, 1] - x0[:, 0], x0[:, 2] - x0[:, 0]], axis=2)
        det = np.linalg.det(dm)
        area = 0.5 * np.abs(det)
        self._tri_inv_dm = np.linalg.inv(dm)

        grad1 = self._tri_inv_dm[:, :, 0]
        grad2 = self._tri_inv_dm[:, :, 1]
        grad0 = -grad1 - grad2

        b = np.zeros((x0.shape[0], 3, 6), dtype=np.float64)
        b[:, 0, 0] = grad0[:, 0]
        b[:, 0, 2] = grad1[:, 0]
        b[:, 0, 4] = grad2[:, 0]
        b[:, 1, 1] = grad0[:, 1]
        b[:, 1, 3] = grad1[:, 1]
        b[:, 1, 5] = grad2[:, 1]
        b[:, 2, 0] = grad0[:, 1]
        b[:, 2, 1] = grad0[:, 0]
        b[:, 2, 2] = grad1[:, 1]
        b[:, 2, 3] = grad1[:, 0]
        b[:, 2, 4] = grad2[:, 1]
        b[:, 2, 5] = grad2[:, 0]

        k_grid = self.k_map.reshape(-1)
        k_nodes = k_grid[self.node_grid_index]
        young = self.fem_scale * (k_nodes[idx0] + k_nodes[idx1] + k_nodes[idx2]) / 3.0
        # Map spring-like k to Young's modulus scale (stiffer for small edge length).
        young = young / max(self.edge_len, 1e-8)
        young = np.clip(young, 1e-8, None)
        nu = self.poisson
        base = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) * 0.5]])
        scale = young / (1.0 - nu * nu)
        d = scale[:, None, None] * base[None, :, :]

        bt = np.transpose(b, (0, 2, 1))
        ke = np.einsum("tij,tjk->tik", bt, d)
        ke = np.einsum("tij,tjk->tik", ke, b)
        ke *= area[:, None, None]
        self._tri_ke = ke

        x0_flat = x0.reshape(-1, 6)
        self._tri_ke_x0 = np.einsum("tij,tj->ti", self._tri_ke, x0_flat)

    def _init_fem_backend(self, device: str):
        device = (device or "cpu").lower()
        if device == "gpu":
            try:
                import cupy as cp
            except Exception as exc:
                raise ImportError(
                    "FEM GPU 计算需要 cupy，请先安装（例如 pip install cupy-cuda11x）。"
                ) from exc
            self._cp = cp
            self._use_gpu = True
            self._cp.cuda.Device().use()
            self._tri_gpu = self._build_fem_gpu_buffers()
            self._force_ext_gpu = cp.zeros(self.force_ext.shape, dtype=cp.float64)
        else:
            self._use_gpu = False

    def _build_fem_gpu_buffers(self):
        if self._tri_indices is None or self._tri_indices.size == 0:
            return None
        cp = self._cp
        tri_idx = cp.asarray(self._tri_indices, dtype=cp.int32)
        tri_gpu = {
            "tri_idx": tri_idx,
            "idx0": tri_idx[:, 0],
            "idx1": tri_idx[:, 1],
            "idx2": tri_idx[:, 2],
            "inv_dm": cp.asarray(self._tri_inv_dm, dtype=cp.float64),
            "ke": cp.asarray(self._tri_ke, dtype=cp.float64),
            "ke_x0": cp.asarray(self._tri_ke_x0, dtype=cp.float64),
        }
        return tri_gpu
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
        cell_shape = (max(self.rows - 1, 0), max(self.cols - 1, 0))
        if not hasattr(self, "removed_cells") or self.removed_cells.shape != cell_shape:
            self.removed_cells = np.zeros(cell_shape, dtype=bool)
            self.removed_cells_version = 0
        elif not hasattr(self, "removed_cells_version"):
            self.removed_cells_version = 0
        self._rebuild_cut_masks_from_removed_cells()

    def set_active_mask(self, mask: np.ndarray, rebuild: bool = True):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.rows, self.cols):
            raise ValueError("active_mask shape must match (rows, cols).")
        self.active_grid = mask
        self.fixed_grid &= self.active_grid
        if rebuild:
            self._rebuild_topology(preserve_state=True)
        if getattr(self, "node_grid_index", None) is not None:
            self._update_pin_from_fixed_grid()

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
            if self.active_grid is not None:
                neigh_active, _ = self._shift_with_mask(self.active_grid, dr, dc)
                valid &= self.active_grid & neigh_active
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

    def _rebuild_cut_masks_from_removed_cells(self):
        if not hasattr(self, "cut_masks") or not hasattr(self, "removed_cells"):
            return
        for off in self.cut_masks:
            self.cut_masks[off].fill(False)
        if self.rows < 2 or self.cols < 2:
            self._build_draw_edges()
            return

        removed = self.removed_cells
        # 水平边：上下两侧单元都被移除才切断
        above = np.zeros((self.rows, self.cols - 1), dtype=bool)
        below = np.zeros((self.rows, self.cols - 1), dtype=bool)
        above[1:, :] = removed
        below[:-1, :] = removed
        cut_h = above & below
        # 边界水平边：只有一侧有单元，单元被移除即切断
        cut_h[0, :] |= removed[0, :]
        cut_h[-1, :] |= removed[-1, :]
        self.cut_masks[(0, 1)][:, :-1] = cut_h
        self.cut_masks[(0, -1)][:, 1:] = cut_h

        # 垂直边：左右两侧单元都被移除才切断
        left = np.zeros((self.rows - 1, self.cols), dtype=bool)
        right = np.zeros((self.rows - 1, self.cols), dtype=bool)
        left[:, 1:] = removed
        right[:, :-1] = removed
        cut_v = left & right
        # 边界垂直边：只有一侧有单元，单元被移除即切断
        cut_v[:, 0] |= removed[:, 0]
        cut_v[:, -1] |= removed[:, -1]
        self.cut_masks[(1, 0)][:-1, :] = cut_v
        self.cut_masks[(-1, 0)][1:, :] = cut_v

        # 对角弹簧：所在单元被移除就切断
        self.cut_masks[(1, 1)][:-1, :-1] = removed
        self.cut_masks[(-1, -1)][1:, 1:] = removed
        self.cut_masks[(1, -1)][:-1, 1:] = removed
        self.cut_masks[(-1, 1)][1:, :-1] = removed
        self._build_draw_edges()

    # ------------------------------------------------------------------ 更新
    def update_soft_object(self, mass: float, ts: float):
        if self._use_gpu:
            self._update_soft_object_gpu(mass, ts)
            return
        if self._tri_indices is None or self._tri_ke is None or ts <= 0:
            return

        x = self.pos.reshape(-1)
        v = self.vel.reshape(-1)
        f_ext = self.force_ext.reshape(-1)

        rot, tri_active = self._compute_fem_rotation()
        f0 = self._compute_f0(rot, tri_active)

        dt2 = ts * ts
        b = mass * (x + ts * v) + dt2 * (f_ext + f0)

        pin_dofs = None
        pin_weight = getattr(self, "pin_weight", None)
        pin_target = getattr(self, "pin_target", None)
        if pin_weight is not None and pin_target is not None:
            if pin_weight.shape[0] == self.pos.shape[0] and pin_target.shape == self.pos.shape:
                if np.any(pin_weight > 0):
                    pin_weight_eff = pin_weight
                    if hasattr(self, "active"):
                        pin_weight_eff = np.where(self.active, pin_weight_eff, 0.0)
                    if self.fixed_mode.size:
                        pin_weight_eff = np.where(self.fixed_mode, 0.0, pin_weight_eff)
                    if np.any(pin_weight_eff > 0):
                        pin_k = self.k_nodes * self.pin_strength
                        pin_k = pin_k * pin_weight_eff
                        pin_dofs = np.repeat(pin_k, 2)
                        b += dt2 * pin_dofs * pin_target.reshape(-1)

        free_nodes = ~self.fixed_mode
        if hasattr(self, "active"):
            free_nodes &= self.active
        free_dofs = np.repeat(free_nodes, 2)
        if not np.any(free_dofs):
            return

        if np.any(~free_dofs):
            x_fixed = x.copy()
            x_fixed[free_dofs] = 0.0
            b -= dt2 * self._apply_k_rot(x_fixed, rot, tri_active)

        b_free = b[free_dofs]
        x_guess = x[free_dofs]

        def matvec(p_free):
            p_full = np.zeros_like(x)
            p_full[free_dofs] = p_free
            y_full = mass * p_full
            y_full += dt2 * self._apply_k_rot(p_full, rot, tri_active)
            if pin_dofs is not None:
                y_full += dt2 * pin_dofs * p_full
            return y_full[free_dofs]

        x_new_free = self._cg_solve(matvec, b_free, x_guess, max_iter=80, tol=1e-6)
        x_new = x.copy()
        x_new[free_dofs] = x_new_free
        x_new = x_new.reshape(-1, 2)

        v_new = (x_new - self.pos) / ts
        damp = 1.0 - self.damping_nodes * ts
        damp = np.clip(damp, 0.0, 1.0)
        v_new *= damp[:, None]

        v_new[self.fixed_mode] = 0.0
        x_new[self.fixed_mode] = self.pos[self.fixed_mode]
        if hasattr(self, "active"):
            v_new[~self.active] = 0.0
            x_new[~self.active] = self.pos[~self.active]
        if self.sleep_eps > 0.0:
            sleep_eps2 = self.sleep_eps * self.sleep_eps
            speed2 = v_new[:, 0] * v_new[:, 0] + v_new[:, 1] * v_new[:, 1]
            v_new[speed2 < sleep_eps2] = 0.0

        vel_old = self.vel
        self.pos_old = self.pos
        self.pos = x_new
        self.vel = v_new
        self.v = self.pos - self.pos_old
        self.acc = (self.vel - vel_old) / ts
        self.force = f_ext.reshape(-1, 2)

    def _update_soft_object_gpu(self, mass: float, ts: float):
        if self._tri_gpu is None or ts <= 0:
            return
        cp = self._cp
        cp.cuda.Device().use()

        x = cp.asarray(self.pos, dtype=cp.float64).reshape(-1)
        v = cp.asarray(self.vel, dtype=cp.float64).reshape(-1)
        if self._force_ext_gpu is None:
            f_ext = cp.asarray(self.force_ext, dtype=cp.float64).reshape(-1)
        else:
            f_ext = self._force_ext_gpu.reshape(-1)

        rot, tri_active = self._compute_fem_rotation_gpu(x)
        f0 = self._compute_f0_gpu(rot, tri_active)

        dt2 = ts * ts
        b = mass * (x + ts * v) + dt2 * (f_ext + f0)

        pin_dofs = None
        pin_weight = getattr(self, "pin_weight", None)
        pin_target = getattr(self, "pin_target", None)
        if pin_weight is not None and pin_target is not None:
            if pin_weight.shape[0] == self.pos.shape[0] and pin_target.shape == self.pos.shape:
                if np.any(pin_weight > 0):
                    pin_weight_gpu = cp.asarray(pin_weight, dtype=cp.float64)
                    if hasattr(self, "active"):
                        pin_weight_gpu = cp.where(cp.asarray(self.active), pin_weight_gpu, 0.0)
                    pin_weight_gpu = cp.where(cp.asarray(self.fixed_mode), 0.0, pin_weight_gpu)
                    pin_k = cp.asarray(self.k_nodes, dtype=cp.float64) * self.pin_strength
                    pin_k = pin_k * pin_weight_gpu
                    pin_dofs = cp.repeat(pin_k, 2)
                    x_pin = cp.asarray(pin_target, dtype=cp.float64).reshape(-1)
                    b += dt2 * pin_dofs * x_pin

        free_nodes = ~self.fixed_mode
        if hasattr(self, "active"):
            free_nodes &= self.active
        free_dofs = np.repeat(free_nodes, 2)
        if not np.any(free_dofs):
            return
        free_dofs_gpu = cp.asarray(free_dofs)

        if np.any(~free_dofs):
            x_fixed = x.copy()
            x_fixed[free_dofs_gpu] = 0.0
            b -= dt2 * self._apply_k_rot_gpu(x_fixed, rot, tri_active)

        b_free = b[free_dofs_gpu]
        x_guess = x[free_dofs_gpu]

        def matvec(p_free):
            p_full = cp.zeros_like(x)
            p_full[free_dofs_gpu] = p_free
            y_full = mass * p_full
            y_full += dt2 * self._apply_k_rot_gpu(p_full, rot, tri_active)
            if pin_dofs is not None:
                y_full += dt2 * pin_dofs * p_full
            return y_full[free_dofs_gpu]

        x_new_free = self._cg_solve_gpu(matvec, b_free, x_guess, max_iter=80, tol=1e-6)
        x_new = x.copy()
        x_new[free_dofs_gpu] = x_new_free
        x_new = x_new.reshape(-1, 2)

        v_new = (x_new - cp.asarray(self.pos)) / ts
        damp = 1.0 - cp.asarray(self.damping_nodes) * ts
        damp = cp.clip(damp, 0.0, 1.0)
        v_new *= damp[:, None]

        v_new = cp.where(cp.asarray(self.fixed_mode)[..., None], 0.0, v_new)
        x_new = cp.where(cp.asarray(self.fixed_mode)[..., None], cp.asarray(self.pos), x_new)
        if hasattr(self, "active"):
            active = cp.asarray(self.active)
            v_new = cp.where(active[..., None], v_new, 0.0)
            x_new = cp.where(active[..., None], x_new, cp.asarray(self.pos))
        if self.sleep_eps > 0.0:
            sleep_eps2 = self.sleep_eps * self.sleep_eps
            speed2 = cp.sum(v_new * v_new, axis=1)
            v_new[speed2 < sleep_eps2] = 0.0

        vel_old = self.vel
        self.pos_old = self.pos
        self.pos = cp.asnumpy(x_new)
        self.vel = cp.asnumpy(v_new)
        self.v = self.pos - self.pos_old
        self.acc = (self.vel - vel_old) / ts
        self.force = cp.asnumpy(f_ext).reshape(-1, 2)

    # ------------------------------------------------------------------ FEM
    def _compute_fem_rotation(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        idx0 = self._tri_indices[:, 0]
        idx1 = self._tri_indices[:, 1]
        idx2 = self._tri_indices[:, 2]
        x_all = self.pos.reshape(-1, 2)
        p0 = x_all[idx0]
        p1 = x_all[idx1]
        p2 = x_all[idx2]

        ds = np.stack([p1 - p0, p2 - p0], axis=2)
        f = np.einsum("tij,tjk->tik", ds, self._tri_inv_dm)
        u, _, vt = np.linalg.svd(f)
        r = np.einsum("tij,tjk->tik", u, vt)
        det = np.linalg.det(r)
        if np.any(det < 0.0):
            flip = det < 0.0
            u[flip, :, -1] *= -1.0
            r = np.einsum("tij,tjk->tik", u, vt)

        return r, None

    def _compute_fem_rotation_gpu(self, x_flat: "cp.ndarray") -> Tuple["cp.ndarray", Optional["cp.ndarray"]]:
        cp = self._cp
        x_all = x_flat.reshape(-1, 2)
        idx0 = self._tri_gpu["idx0"]
        idx1 = self._tri_gpu["idx1"]
        idx2 = self._tri_gpu["idx2"]
        p0 = x_all[idx0]
        p1 = x_all[idx1]
        p2 = x_all[idx2]
        ds = cp.stack([p1 - p0, p2 - p0], axis=2)
        f = cp.einsum("tij,tjk->tik", ds, self._tri_gpu["inv_dm"])

        a = f[:, 0, 0]
        b = f[:, 0, 1]
        c = f[:, 1, 0]
        d = f[:, 1, 1]
        r00 = a + d
        r01 = b - c
        r10 = c - b
        r11 = a + d
        s = cp.sqrt(r00 * r00 + r01 * r01) + 1e-12
        r = cp.empty_like(f)
        r[:, 0, 0] = r00 / s
        r[:, 0, 1] = r01 / s
        r[:, 1, 0] = r10 / s
        r[:, 1, 1] = r11 / s

        return r, None

    def _rotate_vec(self, rot: np.ndarray, vec: np.ndarray, transpose: bool = False) -> np.ndarray:
        vec2 = vec.reshape(-1, 3, 2)
        if transpose:
            rot_use = np.transpose(rot, (0, 2, 1))
        else:
            rot_use = rot
        out = np.einsum("tij,tnj->tni", rot_use, vec2)
        return out.reshape(-1, 6)

    def _rotate_vec_gpu(self, rot: "cp.ndarray", vec: "cp.ndarray", transpose: bool = False) -> "cp.ndarray":
        cp = self._cp
        vec2 = vec.reshape(-1, 3, 2)
        if transpose:
            rot_use = cp.transpose(rot, (0, 2, 1))
        else:
            rot_use = rot
        out = cp.einsum("tij,tnj->tni", rot_use, vec2)
        return out.reshape(-1, 6)

    def _scatter_tri(self, tri_vec: np.ndarray) -> np.ndarray:
        idx0 = self._tri_indices[:, 0]
        idx1 = self._tri_indices[:, 1]
        idx2 = self._tri_indices[:, 2]
        out = np.zeros((self.pos.shape[0], 2), dtype=np.float64)
        np.add.at(out, idx0, tri_vec[:, 0:2])
        np.add.at(out, idx1, tri_vec[:, 2:4])
        np.add.at(out, idx2, tri_vec[:, 4:6])
        return out.reshape(-1)

    def _scatter_tri_gpu(self, tri_vec: "cp.ndarray") -> "cp.ndarray":
        cp = self._cp
        idx0 = self._tri_gpu["idx0"]
        idx1 = self._tri_gpu["idx1"]
        idx2 = self._tri_gpu["idx2"]
        out = cp.zeros((self.pos.shape[0], 2), dtype=cp.float64)
        cp.add.at(out, idx0, tri_vec[:, 0:2])
        cp.add.at(out, idx1, tri_vec[:, 2:4])
        cp.add.at(out, idx2, tri_vec[:, 4:6])
        return out.reshape(-1)

    def _compute_f0(self, rot: np.ndarray, tri_active: Optional[np.ndarray]) -> np.ndarray:
        f0_tri = self._rotate_vec(rot, self._tri_ke_x0, transpose=False)
        if tri_active is not None:
            f0_tri = np.where(tri_active[:, None], f0_tri, 0.0)
        return self._scatter_tri(f0_tri)

    def _compute_f0_gpu(self, rot: "cp.ndarray", tri_active: Optional["cp.ndarray"]) -> "cp.ndarray":
        f0_tri = self._rotate_vec_gpu(rot, self._tri_gpu["ke_x0"], transpose=False)
        if tri_active is not None:
            f0_tri = self._cp.where(tri_active[:, None], f0_tri, 0.0)
        return self._scatter_tri_gpu(f0_tri)

    def _apply_k_rot(self, vec: np.ndarray, rot: np.ndarray, tri_active: Optional[np.ndarray]) -> np.ndarray:
        vec_tri = self._gather_tri(vec)
        v_local = self._rotate_vec(rot, vec_tri, transpose=True)
        w_local = np.einsum("tij,tj->ti", self._tri_ke, v_local)
        w_tri = self._rotate_vec(rot, w_local, transpose=False)
        if tri_active is not None:
            w_tri = np.where(tri_active[:, None], w_tri, 0.0)
        return self._scatter_tri(w_tri)

    def _apply_k_rot_gpu(
        self, vec: "cp.ndarray", rot: "cp.ndarray", tri_active: Optional["cp.ndarray"]
    ) -> "cp.ndarray":
        vec_tri = self._gather_tri_gpu(vec)
        v_local = self._rotate_vec_gpu(rot, vec_tri, transpose=True)
        w_local = self._cp.einsum("tij,tj->ti", self._tri_gpu["ke"], v_local)
        w_tri = self._rotate_vec_gpu(rot, w_local, transpose=False)
        if tri_active is not None:
            w_tri = self._cp.where(tri_active[:, None], w_tri, 0.0)
        return self._scatter_tri_gpu(w_tri)

    def _gather_tri(self, vec: np.ndarray) -> np.ndarray:
        x_all = vec.reshape(-1, 2)
        idx0 = self._tri_indices[:, 0]
        idx1 = self._tri_indices[:, 1]
        idx2 = self._tri_indices[:, 2]
        return np.stack([x_all[idx0], x_all[idx1], x_all[idx2]], axis=1).reshape(-1, 6)

    def _gather_tri_gpu(self, vec: "cp.ndarray") -> "cp.ndarray":
        cp = self._cp
        x_all = vec.reshape(-1, 2)
        idx0 = self._tri_gpu["idx0"]
        idx1 = self._tri_gpu["idx1"]
        idx2 = self._tri_gpu["idx2"]
        return cp.stack([x_all[idx0], x_all[idx1], x_all[idx2]], axis=1).reshape(-1, 6)

    def _cg_solve(self, matvec, b: np.ndarray, x0: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
        x = x0.copy()
        r = b - matvec(x)
        p = r.copy()
        rr = float(np.dot(r, r))
        if rr < tol * tol:
            return x
        for _ in range(max_iter):
            ap = matvec(p)
            denom = float(np.dot(p, ap))
            if denom <= 1e-12:
                break
            alpha = rr / denom
            x += alpha * p
            r -= alpha * ap
            rr_new = float(np.dot(r, r))
            if rr_new < tol * tol:
                break
            beta = rr_new / rr
            p = r + beta * p
            rr = rr_new
        return x

    def _cg_solve_gpu(
        self, matvec, b: "cp.ndarray", x0: "cp.ndarray", max_iter: int, tol: float
    ) -> "cp.ndarray":
        cp = self._cp
        x = x0.copy()
        r = b - matvec(x)
        p = r.copy()
        rr = float(cp.dot(r, r).get())
        if rr < tol * tol:
            return x
        for _ in range(max_iter):
            ap = matvec(p)
            denom = float(cp.dot(p, ap).get())
            if denom <= 1e-12:
                break
            alpha = rr / denom
            x += alpha * p
            r -= alpha * ap
            rr_new = float(cp.dot(r, r).get())
            if rr_new < tol * tol:
                break
            beta = rr_new / rr
            p = r + beta * p
            rr = rr_new
        return x

    def _finalize_cut(self, rebuild: bool):
        if rebuild:
            self._rebuild_topology(preserve_state=True)
            self._pending_rebuild = False
        else:
            self._pending_rebuild = True

    def commit_cuts(self) -> bool:
        if not getattr(self, "_pending_rebuild", False):
            return False
        self._rebuild_topology(preserve_state=True)
        self._pending_rebuild = False
        return True

    def remove_cells_near(self, x: float, y: float, radius: float, rebuild: bool = True) -> bool:
        if radius is None or radius <= 0.0:
            return self.remove_nearest_cell(x, y, rebuild=rebuild)
        if self.rows < 2 or self.cols < 2:
            return False
        cell_shape = (max(self.rows - 1, 0), max(self.cols - 1, 0))
        if not hasattr(self, "removed_cells") or self.removed_cells.shape != cell_shape:
            self.removed_cells = np.zeros(cell_shape, dtype=bool)
            self.removed_cells_version = 0

        if self.cell_nodes is None or self.cell_nodes.size == 0:
            return False

        valid = np.ones(cell_shape, dtype=bool)
        if self.active_grid is not None:
            active = self.active_grid
            valid &= active[:-1, :-1] & active[1:, :-1] & active[:-1, 1:] & active[1:, 1:]
        valid &= ~self.removed_cells
        if not np.any(valid):
            return False

        flat_valid = np.flatnonzero(valid.reshape(-1))
        nodes = self.cell_nodes.reshape(-1, 4)[flat_valid]
        centers = self.pos[nodes].mean(axis=1)
        target = np.array([x, y], dtype=np.float64)
        diff = centers - target
        dist2 = (diff * diff).sum(axis=1)
        radius2 = float(radius) * float(radius)
        hit = dist2 <= radius2
        if not np.any(hit):
            return False
        flat_cells = flat_valid[hit]
        r = (flat_cells // (self.cols - 1)).astype(int)
        c = (flat_cells % (self.cols - 1)).astype(int)
        self.removed_cells[r, c] = True
        self.removed_cells_version = int(getattr(self, "removed_cells_version", 0)) + 1
        self._finalize_cut(rebuild)
        return True

    def remove_cells_along_segment(
        self, x0: float, y0: float, x1: float, y1: float, radius: float, rebuild: bool = True
    ) -> bool:
        if radius is None or radius <= 0.0:
            return self.remove_nearest_cell(x1, y1, rebuild=rebuild)
        if self.rows < 2 or self.cols < 2:
            return False
        cell_shape = (max(self.rows - 1, 0), max(self.cols - 1, 0))
        if not hasattr(self, "removed_cells") or self.removed_cells.shape != cell_shape:
            self.removed_cells = np.zeros(cell_shape, dtype=bool)
            self.removed_cells_version = 0

        if self.cell_nodes is None or self.cell_nodes.size == 0:
            return False

        valid = np.ones(cell_shape, dtype=bool)
        if self.active_grid is not None:
            active = self.active_grid
            valid &= active[:-1, :-1] & active[1:, :-1] & active[:-1, 1:] & active[1:, 1:]
        valid &= ~self.removed_cells
        if not np.any(valid):
            return False

        flat_valid = np.flatnonzero(valid.reshape(-1))
        nodes = self.cell_nodes.reshape(-1, 4)[flat_valid]
        centers = self.pos[nodes].mean(axis=1)
        ax = float(x0)
        ay = float(y0)
        bx = float(x1)
        by = float(y1)
        vx = bx - ax
        vy = by - ay
        seg_len2 = vx * vx + vy * vy
        if seg_len2 < 1e-12:
            return self.remove_cells_near(ax, ay, radius, rebuild=rebuild)
        px = centers[:, 0] - ax
        py = centers[:, 1] - ay
        t = (px * vx + py * vy) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        cx = ax + t * vx
        cy = ay + t * vy
        dx = centers[:, 0] - cx
        dy = centers[:, 1] - cy
        dist2 = dx * dx + dy * dy
        radius2 = float(radius) * float(radius)
        hit = dist2 <= radius2
        if not np.any(hit):
            return False
        flat_cells = flat_valid[hit]
        r = (flat_cells // (self.cols - 1)).astype(int)
        c = (flat_cells % (self.cols - 1)).astype(int)
        self.removed_cells[r, c] = True
        self.removed_cells_version = int(getattr(self, "removed_cells_version", 0)) + 1
        self._finalize_cut(rebuild)
        return True

    def remove_nearest_cell(
        self, x: float, y: float, max_dist: Optional[float] = None, rebuild: bool = True
    ) -> bool:
        if self.rows < 2 or self.cols < 2:
            return False
        cell_shape = (max(self.rows - 1, 0), max(self.cols - 1, 0))
        if not hasattr(self, "removed_cells") or self.removed_cells.shape != cell_shape:
            self.removed_cells = np.zeros(cell_shape, dtype=bool)
            self.removed_cells_version = 0

        if self.cell_nodes is None or self.cell_nodes.size == 0:
            return False

        valid = np.ones(cell_shape, dtype=bool)
        if self.active_grid is not None:
            active = self.active_grid
            valid &= active[:-1, :-1] & active[1:, :-1] & active[:-1, 1:] & active[1:, 1:]
        valid &= ~self.removed_cells
        if not np.any(valid):
            return False

        flat_valid = np.flatnonzero(valid.reshape(-1))
        nodes = self.cell_nodes.reshape(-1, 4)[flat_valid]
        centers = self.pos[nodes].mean(axis=1)
        target = np.array([x, y], dtype=np.float64)
        diff = centers - target
        dist2 = (diff * diff).sum(axis=1)
        best_idx = int(np.argmin(dist2))
        best_dist2 = float(dist2[best_idx])
        if not np.isfinite(best_dist2):
            return False
        if max_dist is not None and best_dist2 > max_dist * max_dist:
            return False

        flat_cell = int(flat_valid[best_idx])
        r = int(flat_cell // (self.cols - 1))
        c = int(flat_cell % (self.cols - 1))
        self.removed_cells[r, c] = True
        self.removed_cells_version = int(getattr(self, "removed_cells_version", 0)) + 1
        self._finalize_cut(rebuild)
        return True

    def get_cut_cell_mask(self) -> Optional[np.ndarray]:
        if self.rows < 2 or self.cols < 2:
            return None
        mask = np.ones((self.rows - 1, self.cols - 1), dtype=bool)
        if self.active_grid is not None:
            active = self.active_grid
            mask &= active[:-1, :-1] & active[1:, :-1] & active[:-1, 1:] & active[1:, 1:]
        if hasattr(self, "removed_cells"):
            if self.removed_cells.shape != mask.shape:
                self.removed_cells = np.zeros(mask.shape, dtype=bool)
                self.removed_cells_version = 0
            mask &= ~self.removed_cells
        return mask

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
        if self.cell_nodes is None or self.cell_nodes.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        edges = set()
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                if not self.cell_active[r, c]:
                    continue
                v00, v10, v01, v11 = self.cell_nodes[r, c]
                for a, b in [(v00, v10), (v10, v11), (v11, v01), (v01, v00)]:
                    if a < 0 or b < 0:
                        continue
                    edge = (a, b) if a < b else (b, a)
                    edges.add(edge)
        if not edges:
            return np.zeros((0, 2), dtype=np.float64)
        coords = np.zeros((len(edges) * 2, 2), dtype=np.float64)
        for i, (a, b) in enumerate(edges):
            coords[i * 2] = self.pos[a]
            coords[i * 2 + 1] = self.pos[b]
        return coords

    def drawSoftObjectContour(self) -> np.ndarray:
        if self.cell_nodes is None or self.cell_nodes.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        edge_count: Dict[Tuple[int, int], int] = {}
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                if not self.cell_active[r, c]:
                    continue
                v00, v10, v01, v11 = self.cell_nodes[r, c]
                for a, b in [(v00, v10), (v10, v11), (v11, v01), (v01, v00)]:
                    if a < 0 or b < 0:
                        continue
                    edge = (a, b) if a < b else (b, a)
                    edge_count[edge] = edge_count.get(edge, 0) + 1
        boundary = [edge for edge, count in edge_count.items() if count == 1]
        if not boundary:
            return np.zeros((0, 2), dtype=np.float64)
        coords = np.zeros((len(boundary) * 2, 2), dtype=np.float64)
        for i, (a, b) in enumerate(boundary):
            coords[i * 2] = self.pos[a]
            coords[i * 2 + 1] = self.pos[b]
        return coords

    def drawSoftObjectPt(self) -> np.ndarray:
        if hasattr(self, "active"):
            return self.pos[self.active]
        return self.pos.reshape(-1, 2)

    def find_closest_node(self, x: float, y: float) -> int:
        if self.pos.size == 0:
            return -1
        target = np.array([x, y], dtype=np.float64)
        diff = self.pos - target
        dist2 = (diff * diff).sum(axis=1)
        if hasattr(self, "active"):
            if not np.any(self.active):
                return -1
            dist2 = np.where(self.active, dist2, np.inf)
        flat_idx = int(np.argmin(dist2))
        if not np.isfinite(dist2[flat_idx]):
            return -1
        return flat_idx

    def toggle_fixed(self, node_idx: int, force: Optional[bool] = None) -> bool:
        if node_idx < 0 or node_idx >= self.pos.shape[0]:
            return False
        grid_idx = int(self.node_grid_index[node_idx])
        r = int(grid_idx // self.cols)
        c = int(grid_idx % self.cols)
        if force is None:
            new_state = not self.fixed_grid[r, c]
        else:
            new_state = bool(force)
        if new_state == self.fixed_grid[r, c]:
            return False
        self.fixed_grid[r, c] = new_state
        mask = self.node_grid_index == grid_idx
        self.fixed_mode[mask] = new_state
        if new_state:
            # Freeze current configuration to avoid sudden constraint shocks.
            self.pos_old = self.pos.copy()
            self.vel.fill(0.0)
            self.acc.fill(0.0)
            self.v.fill(0.0)
            self.force_ext.fill(0.0)
            if self._use_gpu and self._force_ext_gpu is not None:
                self._force_ext_gpu.fill(0.0)
            self.pos_old[mask] = self.pos[mask]
            self.vel[mask] = 0.0
            self.acc[mask] = 0.0
            self.force_ext[mask] = 0.0
        self._update_pin_from_fixed_grid()
        return True

    def toggle_fixed_nearest(self, x: float, y: float, max_dist: Optional[float] = None) -> bool:
        idx = self.find_closest_node(x, y)
        if idx < 0:
            return False
        if max_dist is not None:
            dist = float(np.linalg.norm(self.pos[idx] - np.array([x, y], dtype=np.float64)))
            if dist > max_dist:
                return False
        return self.toggle_fixed(idx)

    # ------------------------------------------------------------------ 辅助接口
    def set_force_ext(self, node_idx: int, fx: float, fy: float):
        if node_idx < 0 or node_idx >= self.pos.shape[0]:
            return
        self.force_ext[node_idx, 0] = fx
        self.force_ext[node_idx, 1] = fy
        if self._use_gpu and self._force_ext_gpu is not None:
            self._force_ext_gpu[node_idx, 0] = fx
            self._force_ext_gpu[node_idx, 1] = fy

    def clear_force_ext(self, node_idx: int):
        if node_idx < 0 or node_idx >= self.pos.shape[0]:
            return
        self.force_ext[node_idx, :] = 0.0
        if self._use_gpu and self._force_ext_gpu is not None:
            self._force_ext_gpu[node_idx, :] = 0.0

    def set_conn_mode(self, r: int, c: int, mode: int):
        self.conn_mode[r, c] = mode
        self._init_conn_masks()
