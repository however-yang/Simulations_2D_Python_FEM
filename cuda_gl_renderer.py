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
        self.space_down = False
        self.mouse_pos: Optional[Tuple[float, float]] = None
        self.selected_idx: Optional[Tuple[int, int]] = None
        self.quit = False
        self.cut_pending = False
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
        x, y = glfw.get_cursor_pos(window)
        wx, wy = self._screen_to_world(x, y)
        if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            if self.so.remove_nearest_cell(wx, wy):
                self.cut_pending = True
            return
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
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
        if key == glfw.KEY_SPACE:
            if action == glfw.PRESS:
                self.space_down = True
            elif action == glfw.RELEASE:
                self.space_down = False
            return
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


def _build_grid_uvs(rows: int, cols: int) -> np.ndarray:
    if cols <= 1:
        u = np.zeros(1, dtype=np.float32)
    else:
        u = np.linspace(0.0, 1.0, cols, dtype=np.float32)
    if rows <= 1:
        v = np.zeros(1, dtype=np.float32)
    else:
        v = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.stack([uu, vv], axis=2).reshape(-1, 2).astype(np.float32)


def _build_triangle_indices(
    rows: int, cols: int, active_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    if rows < 2 or cols < 2:
        return np.zeros((0,), dtype=np.uint32)
    grid = np.arange(rows * cols, dtype=np.uint32).reshape(rows, cols)
    if active_mask is None:
        i0 = grid[:-1, :-1].ravel()
        i1 = grid[:-1, 1:].ravel()
        i2 = grid[1:, :-1].ravel()
        i3 = grid[1:, 1:].ravel()
    else:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != (rows, cols):
            raise ValueError("active_mask shape must match (rows, cols).")
        quad_mask = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
        if not np.any(quad_mask):
            return np.zeros((0,), dtype=np.uint32)
        i0 = grid[:-1, :-1][quad_mask]
        i1 = grid[:-1, 1:][quad_mask]
        i2 = grid[1:, :-1][quad_mask]
        i3 = grid[1:, 1:][quad_mask]
    tris = np.stack([i0, i2, i1, i1, i2, i3], axis=1).ravel()
    return tris.astype(np.uint32)


def _compile_shader(source: str, shader_type: int) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="ignore")
        GL.glDeleteShader(shader)
        raise RuntimeError(f"Shader compile failed: {log}")
    return shader


def _link_program(vertex_src: str, fragment_src: str) -> int:
    vs = _compile_shader(vertex_src, GL.GL_VERTEX_SHADER)
    fs = _compile_shader(fragment_src, GL.GL_FRAGMENT_SHADER)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="ignore")
        GL.glDeleteProgram(program)
        raise RuntimeError(f"Shader link failed: {log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return program


def _make_ortho_matrix(left: float, right: float, bottom: float, top: float) -> np.ndarray:
    rl = right - left
    tb = top - bottom
    if abs(rl) < 1e-12 or abs(tb) < 1e-12:
        return np.eye(4, dtype=np.float32)
    return np.array(
        [
            [2.0 / rl, 0.0, 0.0, -(right + left) / rl],
            [0.0, 2.0 / tb, 0.0, -(top + bottom) / tb],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


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
        active_mask: Optional[np.ndarray] = None,
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
        self.active_mask = active_mask
        self.window = None
        self.vbo = None
        self.uv_vbo = None
        self.grid_uv_vbo = None
        self.ebo = None
        self.texture_id = None
        self.mask_texture_id = None
        self.mask_internal_format = None
        self.mask_format = None
        self.mask_size = np.zeros(2, dtype=np.float32)
        self.mask_version = -1
        self._last_removed_cells = None
        self.shader = None
        self.attrib_pos = None
        self.attrib_uv = None
        self.attrib_grid_uv = None
        self.uniform_mvp = None
        self.uniform_tex = None
        self.uniform_mask = None
        self.uniform_mask_size = None
        self.mvp = None
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
        self.mvp = _make_ortho_matrix(self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])
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

    def _pick_mask_format(self):
        if hasattr(GL, "GL_RED") and hasattr(GL, "GL_R8"):
            return GL.GL_R8, GL.GL_RED
        if hasattr(GL, "GL_LUMINANCE"):
            return GL.GL_LUMINANCE, GL.GL_LUMINANCE
        return GL.GL_ALPHA, GL.GL_ALPHA

    def _init_mask_texture(self):
        if self.rows < 2 or self.cols < 2:
            return
        self.mask_internal_format, self.mask_format = self._pick_mask_format()
        self.mask_size = np.array([self.cols - 1, self.rows - 1], dtype=np.float32)
        mask = np.full((self.rows - 1, self.cols - 1), 255, dtype=np.uint8)
        self.mask_texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.mask_texture_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            self.mask_internal_format,
            self.cols - 1,
            self.rows - 1,
            0,
            self.mask_format,
            GL.GL_UNSIGNED_BYTE,
            mask,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _init_shader_program(self):
        vertex_src = """
        #version 120
        attribute vec2 a_pos;
        attribute vec2 a_uv;
        attribute vec2 a_grid_uv;
        uniform mat4 u_mvp;
        varying vec2 v_uv;
        varying vec2 v_grid_uv;
        void main() {
            v_uv = a_uv;
            v_grid_uv = a_grid_uv;
            gl_Position = u_mvp * vec4(a_pos.xy, 0.0, 1.0);
        }
        """
        fragment_src = """
        #version 120
        uniform sampler2D u_tex;
        uniform sampler2D u_mask;
        uniform vec2 u_mask_size;
        varying vec2 v_uv;
        varying vec2 v_grid_uv;
        void main() {
            vec2 grid = v_grid_uv * u_mask_size;
            vec2 cell = floor(grid);
            vec2 max_cell = u_mask_size - vec2(1.0);
            cell = clamp(cell, vec2(0.0), max_cell);
            vec2 mask_uv = (cell + vec2(0.5)) / u_mask_size;
            float mask = texture2D(u_mask, mask_uv).r;
            if (mask < 0.5) {
                discard;
            }
            gl_FragColor = texture2D(u_tex, v_uv);
        }
        """
        self.shader = _link_program(vertex_src, fragment_src)
        self.attrib_pos = GL.glGetAttribLocation(self.shader, "a_pos")
        self.attrib_uv = GL.glGetAttribLocation(self.shader, "a_uv")
        self.attrib_grid_uv = GL.glGetAttribLocation(self.shader, "a_grid_uv")
        self.uniform_mvp = GL.glGetUniformLocation(self.shader, "u_mvp")
        self.uniform_tex = GL.glGetUniformLocation(self.shader, "u_tex")
        self.uniform_mask = GL.glGetUniformLocation(self.shader, "u_mask")
        self.uniform_mask_size = GL.glGetUniformLocation(self.shader, "u_mask_size")

    def _update_mask_from_so(self, so):
        if self.mask_texture_id is None:
            return
        removed = getattr(so, "removed_cells", None)
        if removed is None:
            return
        if removed.shape != (self.rows - 1, self.cols - 1):
            return
        version = getattr(so, "removed_cells_version", None)
        if version is not None:
            if int(version) == self.mask_version:
                return
        else:
            if self._last_removed_cells is not None and np.array_equal(removed, self._last_removed_cells):
                return
        mask = np.where(removed, 0, 255).astype(np.uint8, copy=False)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.mask_texture_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.cols - 1,
            self.rows - 1,
            self.mask_format,
            GL.GL_UNSIGNED_BYTE,
            mask,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        if version is not None:
            self.mask_version = int(version)
        else:
            self._last_removed_cells = removed.copy()

    def _init_texture_mesh(self):
        if self.rows < 2 or self.cols < 2:
            self.draw_mode = "points"
            return

        self._resize_vbo(self.rows * self.cols)
        uvs = _build_uvs(self.rows, self.cols, repeat=self.texture_repeat, flip_v=True)
        grid_uvs = _build_grid_uvs(self.rows, self.cols)
        indices = _build_triangle_indices(self.rows, self.cols, self.active_mask)
        self.index_count = int(indices.size)
        if self.index_count == 0:
            self.draw_mode = "points"
            return

        self.uv_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.uv_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL.GL_STATIC_DRAW)

        self.grid_uv_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.grid_uv_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, grid_uvs.nbytes, grid_uvs, GL.GL_STATIC_DRAW)

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
        self._init_mask_texture()
        self._init_shader_program()

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
            self._update_mask_from_so(so)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            if self.shader is None or self.texture_id is None or self.mask_texture_id is None:
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                return

            GL.glUseProgram(self.shader)
            if self.uniform_mvp is not None and self.mvp is not None:
                GL.glUniformMatrix4fv(self.uniform_mvp, 1, GL.GL_TRUE, self.mvp)
            if self.uniform_tex is not None:
                GL.glUniform1i(self.uniform_tex, 0)
            if self.uniform_mask is not None:
                GL.glUniform1i(self.uniform_mask, 1)
            if self.uniform_mask_size is not None:
                GL.glUniform2f(
                    self.uniform_mask_size,
                    float(self.mask_size[0]),
                    float(self.mask_size[1]),
                )

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.mask_texture_id)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            if self.attrib_pos is not None and self.attrib_pos >= 0:
                GL.glEnableVertexAttribArray(self.attrib_pos)
                GL.glVertexAttribPointer(self.attrib_pos, 2, GL.GL_FLOAT, False, 0, None)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.uv_vbo)
            if self.attrib_uv is not None and self.attrib_uv >= 0:
                GL.glEnableVertexAttribArray(self.attrib_uv)
                GL.glVertexAttribPointer(self.attrib_uv, 2, GL.GL_FLOAT, False, 0, None)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.grid_uv_vbo)
            if self.attrib_grid_uv is not None and self.attrib_grid_uv >= 0:
                GL.glEnableVertexAttribArray(self.attrib_grid_uv)
                GL.glVertexAttribPointer(self.attrib_grid_uv, 2, GL.GL_FLOAT, False, 0, None)

            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glDrawElements(GL.GL_TRIANGLES, self.index_count, GL.GL_UNSIGNED_INT, None)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

            if self.attrib_pos is not None and self.attrib_pos >= 0:
                GL.glDisableVertexAttribArray(self.attrib_pos)
            if self.attrib_uv is not None and self.attrib_uv >= 0:
                GL.glDisableVertexAttribArray(self.attrib_uv)
            if self.attrib_grid_uv is not None and self.attrib_grid_uv >= 0:
                GL.glDisableVertexAttribArray(self.attrib_grid_uv)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glUseProgram(0)
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
            if self.mask_texture_id is not None:
                GL.glDeleteTextures([self.mask_texture_id])
        except Exception:
            pass
        try:
            if self.shader is not None:
                GL.glDeleteProgram(self.shader)
        except Exception:
            pass
        try:
            if self.vbo is not None:
                GL.glDeleteBuffers(1, [int(self.vbo)])
        except Exception:
            pass
        try:
            if self.uv_vbo is not None:
                GL.glDeleteBuffers(1, [int(self.uv_vbo)])
        except Exception:
            pass
        try:
            if self.grid_uv_vbo is not None:
                GL.glDeleteBuffers(1, [int(self.grid_uv_vbo)])
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
