"""
Experimental CUDA-OpenGL renderer (points/contour) using GLFW + PyOpenGL + PyCUDA.
Requires: pip install pycuda PyOpenGL glfw
Notes:
- Must be created after a CUDA-capable device is available.
- Supports draw_mode 'points' or 'contour'. 'full' falls back to points.
- Interaction: left-click selects nearest node; drag to move cursor; 'q' closes window.
"""
from typing import Optional, Tuple
import numpy as np
import torch
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


class CUDAGLRenderer:
    def __init__(self, rows: int, cols: int, edge_len: float, draw_mode: str = "points", draw_skip: int = 1):
        self.draw_mode = draw_mode if draw_mode in ("points", "contour") else "points"
        self.draw_skip = max(1, draw_skip)
        self.window = None
        self.vbo = None
        self.reg_buffer = None
        self.vertex_capacity = 0
        self.cuda_ctx = None
        self._init_gl(rows, cols, edge_len)
        self._init_cuda()

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
        import pycuda.gl.autoinit  # noqa: F401

        self.cuda_ctx = cudagl.make_default_context()
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

    def update(self, so, selected_idx: Optional[Tuple[int, int]]):
        import pycuda.driver as cuda

        # build canvas on GPU
        if self.draw_mode == "contour":
            canvas = so.drawSoftObjectContour()
        else:
            canvas = so.drawSoftObjectPt()
        if self.draw_skip > 1:
            canvas = canvas[:: self.draw_skip]
        canvas = canvas.contiguous()
        v_count = canvas.shape[0]
        if v_count > self.vertex_capacity:
            self._resize_vbo(int(v_count * 1.2))

        # copy CUDA tensor -> GL buffer via cudaGraphics
        mapped = self.reg_buffer.map()
        ptr, size = mapped.device_ptr_and_size()
        bytes_needed = int(canvas.numel() * canvas.element_size())
        if bytes_needed > size:
            mapped.unmap()
            self._resize_vbo(v_count)
            mapped = self.reg_buffer.map()
            ptr, size = mapped.device_ptr_and_size()
        cuda.memcpy_dtod(ptr, int(canvas.data_ptr()), bytes_needed)
        mapped.unmap()

        # render
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(2, GL.GL_FLOAT, 0, None)
        if self.draw_mode == "contour":
            GL.glDrawArrays(GL.GL_LINE_LOOP, 0, v_count)
        else:
            GL.glDrawArrays(GL.GL_POINTS, 0, v_count)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        try:
            if self.reg_buffer is not None:
                self.reg_buffer.unregister()
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
