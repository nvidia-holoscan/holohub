"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

import logging

import cupy as cp

logger = logging.getLogger(__name__)

# Cache holds CuPy arrays when running GPU-only. (Keyed by wavelength index.)
_HESSIAN_CACHE: dict[int, object] = {}

MAX_REASONABLE_COND_RATIO = 10


def solve_regularized_system(
    data_jacobians,
    data_rhs,
    wavelength_idx: int,
    reg: float,
) -> object:
    """
    Parameters
    ----------
    data_jacobians : NDArray[np.float32]
        Jacobian matrix of shape (features * channels, reconstruction_elements * 2).
    data_rhs : NDArray[np.float32]
        Right-hand side data of shape (features * channels).
    reg : float
        Regularization parameter λ.

    Returns
    -------
    NDArray[np.float32]
        Solution array of shape (reconstruction_elements * 2).
    """
    # add sample dimension
    data_rhs = cp.asarray(data_rhs).reshape(1, -1)

    # Form Hessian and get pre-computed matrix properties
    hessian_reg = _build_regularized_system(
        data_jacobians,
        wavelength_idx,
        reg,
    )

    # Dual formulation: solve smaller system, then back-substitute
    alpha = _solve_square_system(hessian_reg, data_rhs.T)
    solution = data_jacobians.T @ alpha
    return solution.T.squeeze()  # remove sample dimension


def _build_regularized_system(
    data_jacobians,
    wavelength_idx: int,
    reg: float,
) -> object:
    """Build regularized system matrix.

    Parameters
    ----------
    data_jacobians : NDArray[np.float32]
        Jacobian matrix.
    wavelength_idx : int
        Wavelength index for caching.
    reg : float
        Regularization parameter.

    Returns
    -------
    NDArray[np.float32]
        Regularized system matrix
    """
    global _HESSIAN_CACHE
    data_hessian_reg = _HESSIAN_CACHE.get(wavelength_idx)
    if data_hessian_reg is not None:
        logger.debug("Reusing cached Hessian")
        return data_hessian_reg

    # Smaller SPD system: (J J^T + λI) for underdetermined case
    data_hessian = data_jacobians @ data_jacobians.T

    data_hessian_reg = data_hessian + reg * cp.sqrt(cp.linalg.norm(data_hessian)) * cp.eye(
        data_hessian.shape[0], dtype=data_jacobians.dtype
    )

    _HESSIAN_CACHE[wavelength_idx] = data_hessian_reg
    logger.debug("Cached Hessian for reuse")

    return data_hessian_reg


def _solve_square_system(
    A,
    b,
) -> object:
    """
    Parameters
    ----------
    A : NDArray[np.float32]
        Square coefficient matrix of the linear system (typically a Hessian).
    b : NDArray[np.float32]
        Right-hand side vector or matrix.

    Returns
    -------
    NDArray[np.float32]
        Solution to the linear system Ax = b.
    """

    # Validate input
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {A.shape}")
    if b.ndim not in {1, 2} or b.shape[0] != A.shape[0]:
        raise ValueError(f"Incompatible shapes: A={A.shape}, b={b.shape}")
    if not cp.all(cp.isfinite(A)):
        raise ValueError("Matrix A contains non-finite values")
    if not cp.all(cp.isfinite(b)):
        raise ValueError("Vector b contains non-finite values")

    # Ensure symmetry for numerical stability
    A = 0.5 * (A + A.T)

    # Regular inverse
    result = cp.linalg.solve(A, b)
    if not cp.all(cp.isfinite(result)):
        raise RuntimeError(
            "Solver produced non-finite values. This may indicate numerical instability. "
            "Check regularization parameter or input data quality."
        )
    return result
