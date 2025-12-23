"""
Experimental CUDA-OpenGL renderer using GLFW + PyOpenGL + PyCUDA.
Requires: pip install pycuda PyOpenGL glfw
Notes:
- Must be created after a CUDA-capable device is available.
- Supports draw_mode 'points', 'contour', or 'texture' (textured mesh).
- Interaction: left-click selects nearest node; drag to move cursor; 'q' closes window.
"""
from typing import Optional, Tuple
import numpy as np
from OpenGL import GL
import glfw


class CUDAGLInteractor:
    def __init__(self, renderer, so, bounds):
        self.renderer = renderer
        self.so = so
        self.mouse_down = False
        self.mouse_pos: Optional[Tuple[float, float]] = None
        self.selected_idx: Optional[Tuple[int, int]] = None
        self.quit = False
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)
        glfw.set_mouse_button_callback(renderer.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(renderer.window, self._on_cursor)
        glfw.set_key_callback(renderer.window, self._on_key)

    def _screen_to_world(self, x, y):
        xmin, xmax, ymin, ymax = self.bounds
        width, height = glfw.get_framebuffer_size(self.renderer.window)
        x_ndc = 2 * x / max(width, 1) - 1
        y_ndc = 2 * (height - y) / max(height, 1) - 1  # invert y
        wx = xmin + (x_ndc + 1) * 0.5 * (xmax - xmin)
        wy = ymin + (y_ndc + 1) * 0.5 * (ymax - ymin)
        return wx, wy

    def _on_mouse_button(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        x, y = glfw.get_cursor_pos(window)
        wx, wy = self._screen_to_world(x, y)
        if action == glfw.PRESS:
            self.mouse_down = True
            self.mouse_pos = (wx, wy)
            self.selected_idx = self.so.find_closest_node(wx, wy)
        elif action == glfw.RELEASE:
            self.mouse_down = False

    def _on_cursor(self, window, x, y):
        if not self.mouse_down:
            return
        wx, wy = self._screen_to_world(x, y)
        self.mouse_pos = (wx, wy)

    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS and (key == glfw.KEY_Q or key == glfw.KEY_ESCAPE):
            self.quit = True

    def process_events(self):
        glfw.poll_events()
        if glfw.window_should_close(self.renderer.window):
            self.quit = True


def _make_checker_texture(size: int = 256, tiles: int = 8) -> np.ndarray:
    size = max(2, int(size))
    tiles = max(1, int(tiles))
    tile = max(1, size // tiles)
    y = np.arange(size)[:, None] // tile
    x = np.arange(size)[None, :] // tile
    mask = (x + y) % 2
    color_a = np.array([235, 235, 235, 255], dtype=np.uint8)
    color_b = np.array([46, 46, 46, 255], dtype=np.uint8)
    return np.where(mask[..., None] == 0, color_a, color_b)


def _as_uint8_rgba(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    if img.shape[2] == 3:
        alpha = np.full((*img.shape[:2], 1), 255, dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
    elif img.shape[2] != 4:
        raise ValueError("Texture image must have 1, 3 or 4 channels.")

    if img.dtype != np.uint8:
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0) * 255.0
        img = img.astype(np.uint8)
    return img


def _build_uvs(rows: int, cols: int, repeat: int = 1, flip_v: bool = True) -> np.ndarray:
    repeat = max(1, int(repeat))
    if cols <= 1:
        u = np.zeros(1, dtype=np.float32)
    else:
        u = np.linspace(0.0, float(repeat), cols, dtype=np.float32)
    if rows <= 1:
        v = np.zeros(1, dtype=np.float32)
    else:
        v = np.linspace(0.0, float(repeat), rows, dtype=np.float32)
    if flip_v:
        v = v[::-1]
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.stack([uu, vv], axis=2).reshape(-1, 2).astype(np.float32)


def _build_triangle_indices(rows: int, cols: int) -> np.ndarray:
    if rows < 2 or cols < 2:
        return np.zeros((0,), dtype=np.uint32)
    grid = np.arange(rows * cols, dtype=np.uint32).reshape(rows, cols)
    i0 = grid[:-1, :-1].ravel()
    i1 = grid[:-1, 1:].ravel()
    i2 = grid[1:, :-1].ravel()
    i3 = grid[1:, 1:].ravel()
    tris = np.stack([i0, i2, i1, i1, i2, i3], axis=1).ravel()
    return tris.astype(np.uint32)


class CUDAGLRenderer:
    def __init__(
        self,
        rows: int,
        cols: int,
        edge_len: float,
        draw_mode: str = "points",
        draw_skip: int = 1,
        texture_image: Optional[np.ndarray] = None,
        texture_repeat: int = 1,
    ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.edge_len = float(edge_len)
        self.draw_mode = draw_mode if draw_mode in ("points", "contour", "texture") else "points"
        self.draw_skip = max(1, draw_skip)
        if self.draw_mode == "texture":
            self.draw_skip = 1
        self.texture_image = texture_image
        self.texture_repeat = max(1, int(texture_repeat))
        self.window = None
        self.vbo = None
        self.uv_vbo = None
        self.ebo = None
        self.texture_id = None
        self.index_count = 0
        self.reg_buffer = None
        self.vertex_capacity = 0
        self.cuda_ctx = None
        self._init_gl(self.rows, self.cols, self.edge_len)
        self._init_cuda()
        if self.draw_mode == "texture":
            self._init_texture_mesh()

    def _init_gl(self, rows: int, cols: int, edge_len: float):
        if not glfw.init():
            raise RuntimeError("GLFW init failed (required for CUDA-OpenGL renderer).")
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(900, 700, "CUDA-OpenGL Renderer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed.")
        glfw.make_context_current(self.window)
        xmax = max(rows, cols) * edge_len + 0.1
        self.bounds = (-0.1, xmax, -0.1, xmax)
        GL.glViewport(0, 0, 900, 700)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3], -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glPointSize(3.0)

    def _init_cuda(self):
        import pycuda.driver as cuda
        import pycuda.gl as cudagl
        from pycuda.tools import make_default_context

        cuda.init()
        self.cuda_ctx = make_default_context(lambda dev: cudagl.make_context(dev))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.vertex_capacity = 1  # will resize on first update
        self.vbo = GL.glGenBuffers(1)
        self.reg_buffer = None
        self._resize_vbo(1024)

    def _resize_vbo(self, vertex_count: int):
        import pycuda.gl as cudagl
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        buf_size = vertex_count * 2 * 4
        GL.glBufferData(GL.GL_ARRAY_BUFFER, buf_size, None, GL.GL_DYNAMIC_DRAW)
        if self.reg_buffer is not None:
            self.reg_buffer.unregister()
        self.reg_buffer = cudagl.RegisteredBuffer(int(self.vbo))
        self.vertex_capacity = vertex_count

    def _init_texture_mesh(self):
        if self.rows < 2 or self.cols < 2:
            self.draw_mode = "points"
            return

        self._resize_vbo(self.rows * self.cols)
        uvs = _build_uvs(self.rows, self.cols, repeat=self.texture_repeat, flip_v=True)
        indices = _build_triangle_indices(self.rows, self.cols)
        self.index_count = int(indices.size)

        self.uv_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.uv_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL.GL_STATIC_DRAW)

        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

        tex = self.texture_image if self.texture_image is not None else _make_checker_texture()
        tex_u8 = _as_uint8_rgba(tex)
        self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            tex_u8.shape[1],
            tex_u8.shape[0],
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            tex_u8,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    def _upload_positions(self, coords, cuda) -> int:
        if hasattr(coords, "data_ptr") and hasattr(coords, "numel"):
            coords = coords.contiguous()
            v_count = int(coords.shape[0])
            bytes_needed = int(coords.numel() * coords.element_size())
            is_torch = True
        else:
            coords = np.ascontiguousarray(coords, dtype=np.float32)
            v_count = int(coords.shape[0])
            bytes_needed = int(coords.nbytes)
            is_torch = False

        if v_count > self.vertex_capacity:
            self._resize_vbo(int(v_count * 1.2))

        mapped = self.reg_buffer.map()
        ptr, size = mapped.device_ptr_and_size()
        if bytes_needed > size:
            mapped.unmap()
            self._resize_vbo(v_count)
            mapped = self.reg_buffer.map()
            ptr, size = mapped.device_ptr_and_size()

        if is_torch:
            cuda.memcpy_dtod(ptr, int(coords.data_ptr()), bytes_needed)
        else:
            cuda.memcpy_htod(ptr, coords)
        mapped.unmap()
        return v_count

    def update(self, so, selected_idx: Optional[Tuple[int, int]]):
        import pycuda.driver as cuda

        if self.draw_mode == "texture":
            coords = so.pos.reshape(-1, 2)
            self._upload_positions(coords, cuda)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointer(2, GL.GL_FLOAT, 0, None)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.uv_vbo)
            GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
            GL.glTexCoordPointer(2, GL.GL_FLOAT, 0, None)

            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            return

        if self.draw_mode == "contour":
            canvas = so.drawSoftObjectContour()
        else:
            canvas = so.drawSoftObjectPt()
        if self.draw_skip > 1:
            canvas = canvas[:: self.draw_skip]
        v_count = self._upload_positions(canvas, cuda)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(2, GL.GL_FLOAT, 0, None)
        if self.draw_mode == "contour":
            GL.glDrawArrays(GL.GL_LINE_LOOP, 0, v_count)
        else:
            GL.glDrawArrays(GL.GL_POINTS, 0, v_count)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        try:
            if self.reg_buffer is not None:
                self.reg_buffer.unregister()
        except Exception:
            pass
        try:
            if self.texture_id is not None:
                GL.glDeleteTextures([self.texture_id])
        except Exception:
            pass
        try:
            if self.uv_vbo is not None:
                GL.glDeleteBuffers(1, [int(self.uv_vbo)])
        except Exception:
            pass
        try:
            if self.ebo is not None:
                GL.glDeleteBuffers(1, [int(self.ebo)])
        except Exception:
            pass
        if self.cuda_ctx is not None:
            try:
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
            except Exception:
                pass
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
