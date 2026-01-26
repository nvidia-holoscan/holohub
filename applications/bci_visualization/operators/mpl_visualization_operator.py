"""Visualization operator for HbO voxel data on brain surface."""

from __future__ import annotations

import gzip
import logging
import scipy.ndimage
import struct
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from time import perf_counter
from typing import Any, Tuple
from urllib.request import urlretrieve
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray

from holoscan.core import (
    ConditionType,
    ExecutionContext,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)

logger = logging.getLogger(__name__)

# URL to the MNI152 surface mesh
MNI152_SURFACE_URL = (
    "https://raw.githubusercontent.com/neurolabusc/surf-ice/master/sample/mni152_2009.mz3"
)

# Default cache directory for downloaded assets
_CACHE_DIR = Path.home() / ".cache" / "myelin"


def _read_mz3(filepath: Path) -> Tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Read an MZ3 mesh file and return faces and vertices.

    Parameters
    ----------
    filepath : Path
        Path to the MZ3 file.

    Returns
    -------
    faces : NDArray[np.int32]
        Face indices with shape (n_faces, 3).
    vertices : NDArray[np.float32]
        Vertex coordinates with shape (n_vertices, 3).

    Raises
    ------
    ValueError
        If the file is not a valid MZ3 file.
    """
    hdr_bytes = 16

    # First, try to read as raw binary
    with open(filepath, "rb") as f:
        data = f.read(hdr_bytes)
        hdr = struct.unpack_from("<HHIII", data)

    # Check if gzip compressed (magic bytes 0x1f8b) or raw MZ3 (0x4D5A = 23117)
    is_gzip = hdr[0] != 23117

    if is_gzip:
        # Re-open with gzip
        f = gzip.open(filepath, "rb")
        data = f.read(hdr_bytes)
        hdr = struct.unpack_from("<HHIII", data)
        if hdr[0] != 23117:
            f.close()
            raise ValueError(f"Not a valid MZ3 file: {filepath}")
    else:
        f = open(filepath, "rb")
        f.read(hdr_bytes)  # Skip header we already read

    try:
        attr = hdr[1]
        n_face = hdr[2]
        n_vert = hdr[3]
        n_skip = hdr[4]

        is_face = (attr & 1) != 0
        is_vert = (attr & 2) != 0

        if attr > 127:
            raise ValueError("Unable to read future version of MZ3 file")
        if n_vert < 1:
            raise ValueError("Unable to read MZ3 files without vertices")
        if n_face < 1 and is_face:
            raise ValueError("MZ3 files with isFACE must specify NFACE")

        # Read faces
        f.seek(hdr_bytes + n_skip)
        faces = np.zeros((0, 3), dtype=np.int32)
        if is_face:
            face_data = f.read(12 * n_face)  # 3 * 4 bytes per face
            faces_flat = np.array(
                struct.unpack_from(f"<{3 * n_face}I", face_data), dtype=np.int32
            )
            faces = faces_flat.reshape((n_face, 3))

        # Read vertices
        f.seek(hdr_bytes + n_skip + (is_face * n_face * 12))
        vertices = np.zeros((0, 3), dtype=np.float32)
        if is_vert:
            vert_data = f.read(12 * n_vert)  # 3 * 4 bytes per vertex
            verts_flat = np.array(
                struct.unpack_from(f"<{3 * n_vert}f", vert_data), dtype=np.float32
            )
            vertices = verts_flat.reshape((n_vert, 3))
    finally:
        f.close()

    return faces, vertices


def _ensure_surface_mesh(cache_dir: Path | None = None) -> Path:
    """Download the MNI152 surface mesh if not already cached.

    Parameters
    ----------
    cache_dir : Path or None
        Directory to cache the mesh. Defaults to ~/.cache/myelin

    Returns
    -------
    Path
        Path to the cached mesh file.
    """
    if cache_dir is None:
        cache_dir = _CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = cache_dir / "mni152_2009.mz3"

    if not mesh_path.exists():
        logger.info("Downloading MNI152 surface mesh to %s", mesh_path)
        urlretrieve(MNI152_SURFACE_URL, mesh_path)
        logger.info("Download complete")

    return mesh_path


def _load_surface_mesh(
    cache_dir: Path | None = None,
) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Load the MNI152 surface mesh.

    Returns
    -------
    coordinates : NDArray[np.float32]
        Vertex coordinates with shape (n_vertices, 3).
    faces : NDArray[np.int32]
        Face indices with shape (n_faces, 3).
    """
    mesh_path = _ensure_surface_mesh(cache_dir)
    faces, vertices = _read_mz3(mesh_path)
    return vertices, faces


def _decimate_mesh(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
    target_faces: int = 10000,
) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Decimate a mesh to reduce face count for faster rendering.

    Uses uniform random sampling of faces. This is a simple approach
    that works well for visualization purposes.

    Parameters
    ----------
    vertices : NDArray[np.float32]
        Vertex coordinates with shape (n_vertices, 3).
    faces : NDArray[np.int32]
        Face indices with shape (n_faces, 3).
    target_faces : int
        Target number of faces after decimation.

    Returns
    -------
    vertices : NDArray[np.float32]
        Vertex coordinates (may be reduced).
    faces : NDArray[np.int32]
        Decimated face indices.
    """
    n_faces = faces.shape[0]
    if n_faces <= target_faces:
        return vertices, faces

    # Randomly sample faces
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    selected_idx = rng.choice(n_faces, size=target_faces, replace=False)
    selected_faces = faces[selected_idx]

    # Find unique vertices used by selected faces
    unique_verts, inverse = np.unique(selected_faces.ravel(), return_inverse=True)
    new_vertices = vertices[unique_verts]
    new_faces = inverse.reshape(-1, 3).astype(np.int32)

    return new_vertices, new_faces


class MplVisualizationOperator(Operator):
    """Operator that visualizes HbO voxel data on a brain surface.

    This operator projects voxel data onto a decimated brain surface mesh
    and renders it using matplotlib's Poly3DCollection for real-time updates.
    """

    # Target face count for decimated mesh (controls performance vs quality)
    TARGET_FACES = 12000

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
        self._colormap = plt.cm.RdBu_r
        self._initialized_plot: bool = False

    def setup(self, spec: OperatorSpec) -> None:
        """Configure operator inputs."""
        spec.input("hb_voxel_data")
        spec.input("affine_4x4").condition(ConditionType.NONE)

    def _ensure_mesh_loaded(self) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Lazy-load and decimate the surface mesh."""
        if self._surface_mesh is None:
            logger.info("Loading MNI152 surface mesh...")
            vertices, faces = _load_surface_mesh()
            logger.info(
                "Original mesh: %d vertices, %d faces",
                vertices.shape[0],
                faces.shape[0],
            )
            # Decimate for real-time performance
            vertices, faces = _decimate_mesh(vertices, faces, self.TARGET_FACES)
            self._surface_mesh = (vertices, faces)
            logger.info(
                "Decimated mesh: %d vertices, %d faces",
                vertices.shape[0],
                faces.shape[0],
            )
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
        surf_mesh = (coords, faces)

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

        # Normalize to [0, 1] for colormap
        vmin, vmax = np.nanmin(face_values), np.nanmax(face_values)
        if vmax - vmin < 1e-10:
            vmax = vmin + 1e-10
        normalized = (face_values - vmin) / (vmax - vmin)
        face_colors = self._colormap(normalized)

        # Initialize figure and mesh on first frame
        if not self._initialized_plot:
            plt.ion()
            self._fig, self._axes = plt.subplots(
                subplot_kw={"projection": "3d"}, figsize=(10, 8)
            )

            # Build polygon vertices for each face: (n_faces, 3, 3)
            triangles = coords[faces]

            self._poly_collection = Poly3DCollection(
                triangles,
                facecolors=face_colors,
                edgecolors="none",
                linewidths=0,
            )
            self._axes.add_collection3d(self._poly_collection)

            # Set axis limits based on mesh bounds
            self._axes.set_xlim(coords[:, 0].min(), coords[:, 0].max())
            self._axes.set_ylim(coords[:, 1].min(), coords[:, 1].max())
            self._axes.set_zlim(coords[:, 2].min(), coords[:, 2].max())
            self._axes.set_box_aspect([1, 1, 1])
            self._axes.axis("off")
            self._axes.view_init(elev=0, azim=-90)  # Lateral view

            self._initialized_plot = True
            logger.info("Initialized real-time visualization")
        else:
            # Update only face colors (fast path)
            self._poly_collection.set_facecolors(face_colors)

        self._axes.set_title(f"HbO Frame {self._frame_count}")

        # Process GUI events to update display
        self._fig.canvas.flush_events()
