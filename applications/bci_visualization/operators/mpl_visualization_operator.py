"""Visualization operator for HbO voxel data on brain surface."""

from __future__ import annotations

import logging
import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from time import perf_counter, sleep
from typing import Any, Tuple
import matplotlib.pyplot as plt
import multiprocessing as mp

import numpy as np
from numpy.typing import NDArray
from nilearn import datasets, surface

from holoscan.core import (
    ConditionType,
    ExecutionContext,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)

logger = logging.getLogger(__name__)

HB_AVERAGE_WINDOW = 100  # number of frames for running average of min/max


def _load_surface_mesh() -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Load the fsaverage surface mesh using nilearn.

    Returns
    -------
    coordinates : NDArray[np.float32]
        Vertex coordinates with shape (n_vertices, 3).
    faces : NDArray[np.int32]
        Face indices with shape (n_faces, 3).
    """
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage4')

    # Load left and right pial surfaces
    coords_left, faces_left = surface.load_surf_mesh(fsaverage['pial_left'])
    coords_right, faces_right = surface.load_surf_mesh(fsaverage['pial_right'])

    # Combine into single mesh
    combined_coords = np.vstack([coords_left, coords_right])
    combined_faces = np.vstack([faces_left, faces_right + coords_left.shape[0]])

    return combined_coords, combined_faces


class PlotServer(mp.Process):
    def __init__(self, vertices, faces, queue):
        super().__init__(daemon=True)
        self.vertices = vertices
        self.faces = faces
        self.queue = queue

    def run(self):
        # This runs in a fresh process whose main thread owns the GUI loop.
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        plt.ion()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
        fig.canvas.manager.set_window_title("Brain Visualization")
        triangles = self.vertices[self.faces]
        default_colors = np.full((self.faces.shape[0], 4), [0.5, 0.5, 0.5, 1.0])
        poly = Poly3DCollection(triangles, facecolors=default_colors, edgecolors=[0.4, 0.4, 0.4])
        ax.add_collection3d(poly)
        ax.set_xlim(self.vertices[:,0].min(), self.vertices[:,0].max())
        ax.set_ylim(self.vertices[:,1].min(), self.vertices[:,1].max())
        ax.set_zlim(self.vertices[:,2].min(), self.vertices[:,2].max())
        ax.set_box_aspect([1,1,1])
        ax.axis("off")
        ax.view_init(elev=-1, azim=31, roll=25)

        # event loop: listen for arrays of shape (n_faces, 4) (RGBA) or None to exit
        try:
            while True:
                try:
                    item = self.queue.get(timeout=0.1)  # small timeout so GUI stays responsive
                except Exception:
                    # no data, still allow GUI to process events
                    fig.canvas.flush_events()
                    sleep(0.01)
                    continue

                if item is None:
                    break  # shutdown signal

                # item expected to be face_colors (n_faces, 4)
                poly.set_facecolors(item)
                fig.canvas.flush_events()
        finally:
            plt.close(fig)


class MplVisualizationOperator(Operator):
    """Operator that visualizes HbO voxel data on a brain surface.

    This operator projects voxel data onto a fsaverage surface mesh
    and renders it using matplotlib's Poly3DCollection for real-time updates.
    """

    def __init__(
        self,
        *,
        fragment: Any | None = None,
    ) -> None:
        """Initialize the visualization operator.

        Parameters
        ----------
        fragment : Any or None
            Parent fragment (Application) for Holoscan wiring.
        """
        super().__init__(fragment, name=self.__class__.__name__)
        self._last_frame_time: float | None = None
        self._frame_count = 0
        self._surface_mesh: Tuple[NDArray[np.float32], NDArray[np.int32]] | None = None
        self._affine: NDArray[np.float32] | None = None
        self._voxel_coords: NDArray[np.float32] | None = None
        # Matplotlib state for real-time updates
        self._fig: Any = None
        self._axes: Any = None
        self._poly_collection: Poly3DCollection | None = None
        self._colormap = plt.cm.coolwarm
        self._hb_min: list[float] = []
        self._hb_max: list[float] = []

    def setup(self, spec: OperatorSpec) -> None:
        """Configure operator inputs."""
        spec.input("hb_voxel_data")
        spec.input("affine_4x4").condition(ConditionType.NONE)

    def start(self) -> None:
        """Initialize the visualization plot."""
        coords, faces = self._ensure_mesh_loaded()

        # create queue and start plot process
        self._plot_queue = mp.Queue(maxsize=4)
        # send numpy arrays (these are pickled/transferred once)
        self._plot_proc = PlotServer(coords, faces, self._plot_queue)
        self._plot_proc.start()

    def stop(self):
        try:
            self._plot_queue.put(None)  # sentinel to ask process to exit
        except Exception:
            pass
        if hasattr(self, "_plot_proc"):
            self._plot_proc.join(timeout=1.0)

    def _ensure_mesh_loaded(self) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Lazy-load the surface mesh."""
        if self._surface_mesh is None:
            logger.info("Loading  surface mesh...")
            vertices, faces = _load_surface_mesh()
            logger.info(
                "Loaded mesh: %d vertices, %d faces",
                vertices.shape[0],
                faces.shape[0],
            )
            self._surface_mesh = (vertices, faces)
        return self._surface_mesh

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ) -> None:
        """Receive voxel data and render on surface."""
        del op_output, context

        hb_frame = op_input.receive("hb_voxel_data")
        affine = op_input.receive("affine_4x4")

        # Store affine on first frame
        if affine is not None and self._affine is None:
            self._affine = np.asarray(affine)
            logger.info("Received affine matrix with shape %s", self._affine.shape)

        now = perf_counter()
        prev = self._last_frame_time
        self._last_frame_time = now

        if prev is None:
            logger.info(
                "Visualizer received HbO voxel frame with shape %s",
                getattr(hb_frame, "shape", None),
            )
        else:
            logger.debug(
                "Visualizer received frame after %.2f ms",
                (now - prev) * 1000.0,
            )

        self._frame_count += 1
        self._render_frame(hb_frame)

    def _render_frame(self, hb_volume: NDArray[np.float32]) -> None:
        """Render the HbO volume on the brain surface.

        Parameters
        ----------
        hb_volume : NDArray[np.float32]
            3D volume of HbO values.
        """
        if self._affine is None:
            logger.warning("No affine matrix received yet, skipping visualization")
            return

        # Ensure we have a numpy array (handle CuPy arrays from GPU-based operators)
        if hasattr(hb_volume, "get"):
            hb_volume = hb_volume.get()

        # Load the surface mesh
        coords, faces = self._ensure_mesh_loaded()

        # Precompute voxel coordinates for the mesh vertices if not done yet
        if self._voxel_coords is None:
            try:
                inv_affine = np.linalg.inv(self._affine)

                # Transform vertices to voxel space: [x, y, z, 1] . inv_affine.T
                # coords is (N, 3)
                ones = np.ones((coords.shape[0], 1), dtype=coords.dtype)
                vertices_homo = np.hstack((coords, ones))  # (N, 4)

                # Result is (N, 4)
                voxel_coords_homo = vertices_homo @ inv_affine.T

                # Store (3, N) for map_coordinates
                self._voxel_coords = voxel_coords_homo[:, :3].T

                logger.info("Precomputed voxel mapping for %d vertices", coords.shape[0])

            except Exception as e:
                logger.error("Failed to compute voxel coordinates: %s", e)
                return

        try:
            # Map volume data to surface using trilinear interpolation
            # map_coordinates expects (ndim, n_points) coords
            surface_data = scipy.ndimage.map_coordinates(
                hb_volume.astype(np.float32),
                self._voxel_coords,
                order=1,
                mode="constant",
                cval=0.0,
            )
        except Exception as e:
            logger.warning("Visualization mapping failed: %s. Skipping frame.", e)
            return

        # Compute per-face colors by averaging vertex values
        face_values = surface_data[faces].mean(axis=1)

        hb_min = np.percentile(face_values, 2)
        self._hb_min.append(hb_min)
        if len(self._hb_min) > HB_AVERAGE_WINDOW:
            self._hb_min.pop(0)
        hb_min = np.mean(self._hb_min)

        hb_max = np.percentile(face_values, 98)
        self._hb_max.append(hb_max)
        if len(self._hb_max) > HB_AVERAGE_WINDOW:
            self._hb_max.pop(0)
        hb_max = np.mean(self._hb_max)

        # Normalize to [-1, 1] for colormap, centered at 0
        hb_abs_max = max(abs(hb_min), abs(hb_max))
        clipped = np.clip(face_values, -hb_abs_max, hb_abs_max)
        normalized = clipped / hb_abs_max

        # Shift to [0, 1] for colormap
        normalized = (normalized + 1) / 2

        face_colors = self._colormap(normalized)

        try:
            # non-blocking put (drop frame if queue is full)
            self._plot_queue.put_nowait(face_colors)
        except Exception:
            # queue full: drop frame to avoid blocking operator
            logger.debug("Plot queue full — dropping frame")
