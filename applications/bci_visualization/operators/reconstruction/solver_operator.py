"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
from typing import Any

import cupy as cp
from processing.reconstruction.reg_inv import solve_regularized_system
from holoscan.core import (
    ExecutionContext,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)
from .types import NormalizedSolveBatch, SolverResult

logger = logging.getLogger(__name__)

class RegularizedSolverOperator(Operator):
    """Run a regularized inverse solver per wavelength system."""
    REG_DEFAULT = 0.1

    def __init__(
        self,
        *,
        reg: float = REG_DEFAULT,
        use_gpu: bool = False,
        fragment: Any | None = None,
    ) -> None:
        super().__init__(fragment, name=self.__class__.__name__)
        self._reg = reg
        self._use_gpu = use_gpu

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("batch")
        spec.output("result")

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ) -> None:
        batch: NormalizedSolveBatch = op_input.receive("batch")
        cuda_stream = op_input.receive_cuda_stream("batch")

        with cp.cuda.ExternalStream(cuda_stream):
            result = self._solve_batch(batch)
            op_output.emit(result, "result")

    def _solve_batch(self, batch: NormalizedSolveBatch) -> SolverResult:
        num_wavelengths = len(batch.systems)
        num_significant_voxels = int(batch.idxs_significant_voxels.size)
        num_cols_expected = batch.num_absorbers * num_significant_voxels

        # GPU-only: always use CuPy.
        result = cp.zeros(
            (num_cols_expected, num_wavelengths),
            dtype=batch.systems[0].jacobian.dtype,
        )
        for wavelength_idx, system in enumerate(batch.systems):
            assert system.rhs.ndim == 1
            assert system.jacobian.shape[1] == num_cols_expected

            solution = solve_regularized_system(
                system.jacobian,
                system.rhs,
                wavelength_idx,
                reg=self._reg,
            )

            solution *= system.background

            assert solution.shape == (num_cols_expected,)
            result[:, wavelength_idx] = solution

        # Reshape result to separate absorbers into mua/musp
        reshaped = result.reshape(
            (-1, batch.num_absorbers, num_wavelengths),
            order="F",
        )
        data_mua = reshaped[:, 0, :]
        data_musp = reshaped[:, 1, :]

        return SolverResult(
            data_mua=data_mua,
            data_musp=data_musp,
            idxs_significant_voxels=batch.idxs_significant_voxels,
            num_full_voxels=batch.num_full_voxels,
            wavelengths=batch.wavelengths,
            voxel_metadata=batch.voxel_metadata,
        )