from __future__ import annotations

import logging
from typing import Any

import numpy as np

from processing.reconstruction import REG_DEFAULT, RESHAPING_ORDER, gpu
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
        del context

        batch: NormalizedSolveBatch = op_input.receive("batch")
        result = self._solve_batch(batch)
        op_output.emit(result, "result")

    def _solve_batch(self, batch: NormalizedSolveBatch) -> SolverResult:
        num_wavelengths = len(batch.systems)
        num_significant_voxels = int(batch.idxs_significant_voxels.size)
        num_cols_expected = batch.num_absorbers * num_significant_voxels

        xp = gpu.get_array_module(self._use_gpu)[0]  # either cupy or numpy
        result = xp.zeros(
            (num_cols_expected, num_wavelengths),
            dtype=batch.systems[0].jacobian.dtype,
        )
        for wavelength_idx, system in enumerate(batch.systems):
            assert system.rhs.ndim == 1
            assert system.jacobian.shape[1] == num_cols_expected

            solution = solve_regularized_system(
                xp,
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
            order=RESHAPING_ORDER,
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