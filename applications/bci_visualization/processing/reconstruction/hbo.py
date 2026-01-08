"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, NamedTuple, Tuple

import cupy as cp

class ExtinctionCoefficient(NamedTuple):
    """
    Matches the row structure of the extinction coefficient CSV asset.
    Provides the molar extinction coefficients for various chromophores at specific wavelengths.
    """
    Wavelength: int
    HbO: float
    deoxyHb: float
    Water: float
    Lipids: float
    LuTex: float
    GdTex: float

    @classmethod
    def from_csv(cls, path: Path) -> Dict[int, ExtinctionCoefficient]:
        def _parse_wavelength(value: str) -> int:
            # Some datasets store wavelength as scientific notation (e.g. "6.0000000e+02").
            # Parse as float then round to nearest integer nm.
            return int(round(float(value)))

        return {
            _parse_wavelength(row["Wavelength"]): cls(
                Wavelength=_parse_wavelength(row["Wavelength"]),
                HbO=float(row["HbO"]),
                deoxyHb=float(row["deoxyHb"]),
                Water=float(row["Water"]),
                Lipids=float(row["Lipids"]),
                LuTex=float(row["LuTex"]),
                GdTex=float(row["GdTex"]),
            )
            for row in csv.DictReader(open(path))
        }

    def get_oxy_deoxy_coefficients(self) -> cp.ndarray:
        return cp.array(
            [
                self.HbO,
                self.deoxyHb,
            ],
            dtype=cp.float32,
        )


class HbO:
    def __init__(self, coefficients: Dict[int, ExtinctionCoefficient], use_gpu: bool = False) -> None:
        self._coefficients = coefficients
        self._cached_coefficients: cp.ndarray | None = None
        self._use_gpu = use_gpu

    def _get_molar_extinction_coefficients(self, wavelength: int) -> ExtinctionCoefficient:
        """Get the molar extinction coefficients for a specific wavelength. These have been converted to µa values and
        interpolated to achieve 1 nm wavelength resolution.

        These values for the molar extinction coefficients e are in [m^-1 / µM], and were originally compiled by
        Scott Prahl (prahl@ece.ogi.edu) using data from:
        W. B. Gratzer, Med. Res. Council Labs, Holly Hill, London
        N. Kollias, Wellman Laboratories, Harvard Medical School, Boston
        https://omlc.org/spectra/hemoglobin/summary.html

        Parameters
        ----------
        wavelength : int
            Wavelength in nanometers.

        Returns
        -------
        NDArray[np.float32]
            Extinction coefficients in [m^-1 / µM] for oxy and deoxy-hemoglobin, water, lipids, LuTex, and GdTex.
        """

        coefficient = self._coefficients.get(round(wavelength))
        if coefficient is None:
            raise ValueError(
                f"No entry found for {wavelength} nm. Please enter a valid integer wavelength between 600-1000 nm."
            )

        return coefficient

    def convert_mua_to_hb(
        self,
        data_mua: cp.ndarray,
        wavelengths: tuple,
        idxs_significant_voxels: cp.ndarray,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Converts mua to Hb in voxel space.

        Parameters
        ----------
        data_mua : NDArray[np.float32]
            µa reconstructed values in voxel space.
        wavelengths : tuple
            Wavelengths (nm).
        idxs_significant_voxels
            Indices of significant voxels over the threshold of
            max sensitivity (jacobian) across voxels.

        Returns
        -------
        data_hbo : NDArray[np.float32]
            Oxygenated hemoglobin in voxel space (µM).
        data_hbr : NDArray[np.float32]
            Deoxygenated hemoglobin in voxel space (µM).
        """
        num_voxels, _ = data_mua.shape

        if self._cached_coefficients is None:
            self._cached_coefficients = cp.asarray(
                [
                    self._get_molar_extinction_coefficients(wavelength).get_oxy_deoxy_coefficients()
                    for wavelength in wavelengths
                ]
            )

        idxs_significant_voxels = cp.asarray(idxs_significant_voxels)
        sample_mua = cp.zeros((len(wavelengths), num_voxels))
        sample_mua[:, idxs_significant_voxels] = data_mua[idxs_significant_voxels, :].T
        sample_hb = cp.linalg.solve(self._cached_coefficients, sample_mua)

        assert sample_hb.shape == (len(wavelengths), num_voxels)
        data_hbo = sample_hb[0]
        data_hbr = sample_hb[1]
        return data_hbo, data_hbr
