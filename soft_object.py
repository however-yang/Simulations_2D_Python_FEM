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
    ):
        self.cols = int(cols)
        self.rows = int(rows)
        self.edge_len = float(edge_len)
        self.k_map = np.full((self.rows, self.cols), float(k), dtype=np.float64)
        self.damping_map = np.full((self.rows, self.cols), float(damping), dtype=np.float64)
        self.fem_scale = max(0.0, float(fem_scale))
        self.poisson = float(np.clip(poisson, 0.0, 0.49))
        self._init_state(pt_fixed_idx)
        self._init_fem_mesh()
        self._use_gpu = False
        self._cp = None
        self._tri_gpu = None
        self._init_fem_backend(fem_device)
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

    def _init_fem_mesh(self):
        if self.rows < 2 or self.cols < 2:
            self._tri_indices = None
            self._tri_cell_flat = None
            self._tri_inv_dm = None
            self._tri_ke = None
            self._tri_ke_x0 = None
            self._tri_x0 = None
            return

        grid = np.arange(self.rows * self.cols, dtype=int).reshape(self.rows, self.cols)
        v00 = grid[:-1, :-1]
        v10 = grid[1:, :-1]
        v01 = grid[:-1, 1:]
        v11 = grid[1:, 1:]
        tri1 = np.stack([v00, v10, v11], axis=2).reshape(-1, 3)
        tri2 = np.stack([v00, v11, v01], axis=2).reshape(-1, 3)
        self._tri_indices = np.vstack([tri1, tri2])

        cell_r = np.repeat(np.arange(self.rows - 1), self.cols - 1)
        cell_c = np.tile(np.arange(self.cols - 1), self.rows - 1)
        cell_flat = cell_r * (self.cols - 1) + cell_c
        self._tri_cell_flat = np.repeat(cell_flat, 2)

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

        x0_all = self.pos_init.reshape(-1, 2)
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

        k_nodes = self.k_map.reshape(-1)
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
            self._tri_gpu = self._build_fem_gpu_buffers()
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
            "tri_cell_flat": cp.asarray(self._tri_cell_flat, dtype=cp.int32),
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

        free_nodes = ~self.fixed_mode
        if hasattr(self, "active"):
            free_nodes &= self.active
        free_dofs = np.repeat(free_nodes.ravel(), 2)
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
            return y_full[free_dofs]

        x_new_free = self._cg_solve(matvec, b_free, x_guess, max_iter=80, tol=1e-6)
        x_new = x.copy()
        x_new[free_dofs] = x_new_free
        x_new = x_new.reshape(self.rows, self.cols, 2)

        v_new = (x_new - self.pos) / ts
        damp = 1.0 - self.damping_map * ts
        damp = np.clip(damp, 0.0, 1.0)
        v_new *= damp[..., None]

        v_new[self.fixed_mode] = 0.0
        x_new[self.fixed_mode] = self.pos[self.fixed_mode]
        if hasattr(self, "active"):
            v_new[~self.active] = 0.0
            x_new[~self.active] = self.pos[~self.active]

        vel_old = self.vel
        self.pos_old = self.pos
        self.pos = x_new
        self.vel = v_new
        self.v = self.pos - self.pos_old
        self.acc = (self.vel - vel_old) / ts
        self.force = f_ext.reshape(self.rows, self.cols, 2)

    def _update_soft_object_gpu(self, mass: float, ts: float):
        if self._tri_gpu is None or ts <= 0:
            return
        cp = self._cp
        cp.cuda.Device().use()

        x = cp.asarray(self.pos, dtype=cp.float64).reshape(-1)
        v = cp.asarray(self.vel, dtype=cp.float64).reshape(-1)
        f_ext = cp.asarray(self.force_ext, dtype=cp.float64).reshape(-1)

        rot, tri_active = self._compute_fem_rotation_gpu(x)
        f0 = self._compute_f0_gpu(rot, tri_active)

        dt2 = ts * ts
        b = mass * (x + ts * v) + dt2 * (f_ext + f0)

        free_nodes = ~self.fixed_mode
        if hasattr(self, "active"):
            free_nodes &= self.active
        free_dofs = np.repeat(free_nodes.ravel(), 2)
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
            return y_full[free_dofs_gpu]

        x_new_free = self._cg_solve_gpu(matvec, b_free, x_guess, max_iter=80, tol=1e-6)
        x_new = x.copy()
        x_new[free_dofs_gpu] = x_new_free
        x_new = x_new.reshape(self.rows, self.cols, 2)

        v_new = (x_new - cp.asarray(self.pos)) / ts
        damp = 1.0 - cp.asarray(self.damping_map) * ts
        damp = cp.clip(damp, 0.0, 1.0)
        v_new *= damp[..., None]

        v_new = cp.where(cp.asarray(self.fixed_mode)[..., None], 0.0, v_new)
        x_new = cp.where(cp.asarray(self.fixed_mode)[..., None], cp.asarray(self.pos), x_new)
        if hasattr(self, "active"):
            active = cp.asarray(self.active)
            v_new = cp.where(active[..., None], v_new, 0.0)
            x_new = cp.where(active[..., None], x_new, cp.asarray(self.pos))

        vel_old = self.vel
        self.pos_old = self.pos
        self.pos = cp.asnumpy(x_new)
        self.vel = cp.asnumpy(v_new)
        self.v = self.pos - self.pos_old
        self.acc = (self.vel - vel_old) / ts
        self.force = cp.asnumpy(f_ext).reshape(self.rows, self.cols, 2)

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

        tri_active = None
        if self.rows > 1 and self.cols > 1:
            cell_mask = np.ones((self.rows - 1, self.cols - 1), dtype=bool)
            if hasattr(self, "active"):
                active = self.active
                cell_mask &= active[:-1, :-1] & active[:-1, 1:] & active[1:, :-1] & active[1:, 1:]
            if hasattr(self, "removed_cells"):
                cell_mask &= ~self.removed_cells
            tri_active = cell_mask.reshape(-1)[self._tri_cell_flat]
        return r, tri_active

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

        tri_active = None
        if self.rows > 1 and self.cols > 1:
            cell_mask = np.ones((self.rows - 1, self.cols - 1), dtype=bool)
            if hasattr(self, "active"):
                active = self.active
                cell_mask &= active[:-1, :-1] & active[:-1, 1:] & active[1:, :-1] & active[1:, 1:]
            if hasattr(self, "removed_cells"):
                cell_mask &= ~self.removed_cells
            tri_active = cp.asarray(cell_mask.reshape(-1)[self._tri_cell_flat], dtype=cp.bool_)
        return r, tri_active

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
        out = np.zeros((self.rows * self.cols, 2), dtype=np.float64)
        np.add.at(out, idx0, tri_vec[:, 0:2])
        np.add.at(out, idx1, tri_vec[:, 2:4])
        np.add.at(out, idx2, tri_vec[:, 4:6])
        return out.reshape(-1)

    def _scatter_tri_gpu(self, tri_vec: "cp.ndarray") -> "cp.ndarray":
        cp = self._cp
        idx0 = self._tri_gpu["idx0"]
        idx1 = self._tri_gpu["idx1"]
        idx2 = self._tri_gpu["idx2"]
        out = cp.zeros((self.rows * self.cols, 2), dtype=cp.float64)
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

    def remove_nearest_cell(self, x: float, y: float, max_dist: Optional[float] = None) -> bool:
        if self.rows < 2 or self.cols < 2:
            return False
        cell_shape = (max(self.rows - 1, 0), max(self.cols - 1, 0))
        if not hasattr(self, "removed_cells") or self.removed_cells.shape != cell_shape:
            self.removed_cells = np.zeros(cell_shape, dtype=bool)
            self._rebuild_cut_masks_from_removed_cells()

        centers = (
            self.pos[:-1, :-1]
            + self.pos[1:, :-1]
            + self.pos[:-1, 1:]
            + self.pos[1:, 1:]
        ) * 0.25
        valid = np.ones(cell_shape, dtype=bool)
        if hasattr(self, "active"):
            active = self.active
            valid &= active[:-1, :-1] & active[1:, :-1] & active[:-1, 1:] & active[1:, 1:]
        valid &= ~self.removed_cells
        if not np.any(valid):
            return False

        target = np.array([x, y], dtype=np.float64)
        diff = centers - target
        dist2 = (diff * diff).sum(axis=2)
        dist2 = np.where(valid, dist2, np.inf)
        flat_idx = int(np.argmin(dist2))
        best_dist2 = float(dist2.ravel()[flat_idx])
        if not np.isfinite(best_dist2):
            return False
        if max_dist is not None and best_dist2 > max_dist * max_dist:
            return False

        r = int(flat_idx // (self.cols - 1))
        c = int(flat_idx % (self.cols - 1))
        self.removed_cells[r, c] = True
        self.removed_cells_version = int(getattr(self, "removed_cells_version", 0)) + 1
        self._rebuild_cut_masks_from_removed_cells()
        return True

    def get_cut_cell_mask(self) -> Optional[np.ndarray]:
        if self.rows < 2 or self.cols < 2:
            return None
        mask = np.ones((self.rows - 1, self.cols - 1), dtype=bool)
        if hasattr(self, "active"):
            active = self.active
            mask &= active[:-1, :-1] & active[1:, :-1] & active[:-1, 1:] & active[1:, 1:]
        if hasattr(self, "removed_cells"):
            if self.removed_cells.shape != mask.shape:
                self.removed_cells = np.zeros(mask.shape, dtype=bool)
                self.removed_cells_version = 0
                self._rebuild_cut_masks_from_removed_cells()
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

    def toggle_fixed(self, r: int, c: int, force: Optional[bool] = None) -> bool:
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        if hasattr(self, "active") and not self.active[r, c]:
            return False
        if force is None:
            new_state = not self.fixed_mode[r, c]
        else:
            new_state = bool(force)
        if new_state == self.fixed_mode[r, c]:
            return False
        self.fixed_mode[r, c] = new_state
        if new_state:
            # Freeze current configuration to avoid sudden constraint shocks.
            self.pos_old = self.pos.copy()
            self.vel.fill(0.0)
            self.acc.fill(0.0)
            self.v.fill(0.0)
            self.force_ext.fill(0.0)
            self.pos_old[r, c] = self.pos[r, c]
            self.vel[r, c, :] = 0.0
            self.acc[r, c, :] = 0.0
            self.force_ext[r, c, :] = 0.0
        return True

    def toggle_fixed_nearest(self, x: float, y: float, max_dist: Optional[float] = None) -> bool:
        r, c = self.find_closest_node(x, y)
        if r < 0 or c < 0:
            return False
        if max_dist is not None:
            dist = float(np.linalg.norm(self.pos[r, c] - np.array([x, y], dtype=np.float64)))
            if dist > max_dist:
                return False
        return self.toggle_fixed(r, c)

    # ------------------------------------------------------------------ 辅助接口
    def set_force_ext(self, r: int, c: int, fx: float, fy: float):
        self.force_ext[r, c, 0] = fx
        self.force_ext[r, c, 1] = fy

    def clear_force_ext(self, r: int, c: int):
        self.force_ext[r, c, :] = 0.0

    def set_conn_mode(self, r: int, c: int, mode: int):
        self.conn_mode[r, c] = mode
        self._init_conn_masks()
