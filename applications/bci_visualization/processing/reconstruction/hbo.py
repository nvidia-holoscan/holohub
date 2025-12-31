from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from .gpu import get_array_module

class ExtinctionCoefficient(NamedTuple):
    Wavelength: float
    HbO: float
    deoxyHb: float
    Water: float
    Lipids: float
    LuTex: float
    GdTex: float

    @classmethod
    def from_csv(cls, path: Path) -> list[ExtinctionCoefficient]:
        return [
            cls(
                Wavelength=float(row["Wavelength"]),
                HbO=float(row["HbO"]),
                deoxyHb=float(row["deoxyHb"]),
                Water=float(row["Water"]),
                Lipids=float(row["Lipids"]),
                LuTex=float(row["LuTex"]),
                GdTex=float(row["GdTex"]),
            )
            for row in csv.DictReader(open(path))
        ]

    def get_oxy_deoxy_coefficients(self) -> NDArray[np.float32]:
        return np.array(
            [
                self.HbO,
                self.deoxyHb,
            ],
            dtype=np.float32,
        )


class HbO:
    def __init__(self, coefficients: list[ExtinctionCoefficient], use_gpu: bool = False) -> None:
        self._coefficients = coefficients
        self._cached_coefficients: NDArray[np.float32] | None = None
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

        wavelength_rows = [ext for ext in self._coefficients if ext.Wavelength == round(wavelength)]
        if len(wavelength_rows) == 0:
            raise ValueError(
                f"No entry found for {wavelength} nm. Please enter a valid integer wavelength between 600-1000 nm."
            )
        if len(wavelength_rows) > 1:
            raise RuntimeError(
                f"Multiple entries found for {wavelength} nm. Please correct the dataset."
            )

        return wavelength_rows[0]

    def convert_mua_to_hb(
        self,
        data_mua: NDArray[np.float32],
        wavelengths: tuple,
        idxs_significant_voxels: NDArray[np.int_],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Converts mua to Hb in voxel space.

        Parameters
        ----------
        data_mua : NDArray[np.float32]
            µa reconstructed values in voxel space.
        wavelengths : tuple
            Wavelengths (nm).
        idxs_significant_voxels
            Indices of significant voxels.

        Returns
        -------
        data_hbo : NDArray[np.float32]
            Oxygenated hemoglobin in voxel space.
        data_hbr : NDArray[np.float32]
            Deoxygenated hemoglobin in voxel space.
        """
        num_voxels, _ = data_mua.shape

        xp = get_array_module(self._use_gpu)[0]  # either cupy or numpy

        if self._cached_coefficients is None:
            self._cached_coefficients = xp.asarray(
                [
                    self._get_molar_extinction_coefficients(wavelength).get_oxy_deoxy_coefficients()
                    for wavelength in wavelengths
                ]
            )

        idxs_significant_voxels = xp.asarray(idxs_significant_voxels)
        sample_mua = xp.zeros((len(wavelengths), num_voxels))
        sample_mua[:, idxs_significant_voxels] = data_mua[idxs_significant_voxels, :].T
        sample_hb = xp.linalg.solve(self._cached_coefficients, sample_mua)

        assert sample_hb.shape == (len(wavelengths), num_voxels)
        data_hbo = sample_hb[0]
        data_hbr = sample_hb[1]
        return data_hbo, data_hbr
