import logging
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, cast

import h5py
import numpy as np

from streams.base_nirs import ChannelInfo

from . import dist3d
from .base_nirs import BaseNirsStream

logger = logging.getLogger(__name__)

NUM_MOMENTS = 3
NUM_WAVELENGTHS = 2


class SNIRFChannel(NamedTuple):
    moment: int
    wavelength: int
    source_module: int
    source_number: int
    detector_module: int
    detector_number: int
    sds: float


class SNIRFStream(BaseNirsStream):
    def __init__(self, snirf_file: Path | str) -> None:
        self._snirf_file_path = Path(snirf_file)
        if not self._snirf_file_path.exists():
            raise FileNotFoundError(f"SNIRF file '{snirf_file}' does not exist")

    def start(self) -> None:
        self._snirf_file = h5py.File(self._snirf_file_path, "r")

        self._channels = self._get_channels()
        self._unique_channels = [
            ch for ch in self._channels if ch.moment == 0 and ch.wavelength == 0
        ]
        print("Got {} unique channels".format(len(self._unique_channels)))

    def get_channels(self) -> ChannelInfo:
        return ChannelInfo(
            source_module=np.array([ch.source_module for ch in self._unique_channels]),
            source_number=np.array([ch.source_number for ch in self._unique_channels]),
            detector_module=np.array([ch.detector_module for ch in self._unique_channels]),
            detector_number=np.array([ch.detector_number for ch in self._unique_channels]),
            sds=np.array([ch.sds for ch in self._unique_channels]),
        )

    def _get_channels(self) -> List[SNIRFChannel]:
        source_pos_3d: List[np.ndarray] = self._snirf_file["nirs"]["probe"]["sourcePos3D"][()]  # type: ignore
        detector_pos_3d: List[np.ndarray] = self._snirf_file["nirs"]["probe"]["detectorPos3D"][()]  # type: ignore

        source_labels: List[bytes] = self._snirf_file["nirs"]["probe"]["sourceLabels"][()]  # type: ignore
        detector_labels: List[bytes] = self._snirf_file["nirs"]["probe"]["detectorLabels"][()]  # type: ignore

        source_pos_3d_map = {}
        for sourceIdx, sourceLabel in enumerate(source_labels):
            m, s = sourceLabel.decode().split("S")
            source_pos_3d_map[(int(m.replace("M", "")), int(s))] = source_pos_3d[sourceIdx]

        detector_pos_3d_map = {}
        for detectorIdx, detectorLabel in enumerate(detector_labels):
            m, d = detectorLabel.decode().split("D")
            detector_pos_3d_map[(int(m.replace("M", "")), int(d))] = detector_pos_3d[detectorIdx]

        moments = self._snirf_file["nirs"]["probe"]["momentOrders"][()]  # type: ignore
        data1 = cast(h5py.Dataset, self._snirf_file["nirs"]["data1"])  # type: ignore
        channel_keys = [key for key in data1 if key.startswith("measurementList")]
        # Sort channel keys numerically (e.g., measurementList1, measurementList2, ..., measurementList10)
        # to match the column order in dataTimeSeries
        channel_keys.sort(key=lambda x: int(x.replace("measurementList", "")))
        channels: List[SNIRFChannel] = []
        for channel_key in channel_keys:
            channel = cast(h5py.Dataset, data1[channel_key])
            source_module, source = (
                source_labels[channel["sourceIndex"][()] - 1].decode().replace("M", "").split("S")
            )
            detector_module, detector = (
                detector_labels[channel["detectorIndex"][()] - 1]
                .decode()
                .replace("M", "")
                .split("D")
            )
            channels.append(
                SNIRFChannel(
                    moment=int(moments[channel["dataTypeIndex"][()] - 1]),  # type: ignore
                    wavelength=int(channel["wavelengthIndex"][()] - 1),
                    source_module=int(source_module),
                    source_number=int(source),
                    detector_module=int(detector_module),
                    detector_number=int(detector),
                    sds=dist3d(
                        *source_pos_3d_map[(int(source_module), int(source))],
                        *detector_pos_3d_map[(int(detector_module), int(detector))],
                    ),
                )
            )

        return channels

    def stream_nirs(self) -> Iterator[np.ndarray]:
        data1 = cast(h5py.Dataset, self._snirf_file["nirs"]["data1"])  # type: ignore
        times: np.ndarray = data1["time"][()]
        data: np.ndarray = data1["dataTimeSeries"][()]

        unique_channel_lut = {
            (ch.source_module, ch.source_number, ch.detector_module, ch.detector_number): idx
            for idx, ch in enumerate(self._unique_channels)
        }
        channel_idxs: Dict[int, Dict[int, Dict[str, List[int]]]] = {}
        for moment in range(NUM_MOMENTS):
            channel_idxs[moment] = {}
            for wavelength in range(NUM_WAVELENGTHS):
                channel_order = [
                    (
                        idx,
                        unique_channel_lut.get(
                            (
                                ch.source_module,
                                ch.source_number,
                                ch.detector_module,
                                ch.detector_number,
                            ),
                            -1,
                        ),
                    )
                    for idx, ch in enumerate(self._channels)
                    if ch.moment == moment and ch.wavelength == wavelength
                ]
                channel_idxs[moment][wavelength] = {
                    "snirf_channel_idxs": [idx for idx, _ in channel_order],
                    "unique_channel_idxs": [uniq_idx for _, uniq_idx in channel_order],
                }

        print("Streaming {} samples from SNIRF".format(len(data)))
        for ts, sample in zip(times, data):
            # sample is shape (n_channels,)
            # send (n_moments, n_unique_channels, n_wavelengths)
            to_send = np.full((NUM_MOMENTS, len(self._unique_channels), NUM_WAVELENGTHS), np.nan)
            for moment in range(NUM_MOMENTS):
                for wavelength in range(NUM_WAVELENGTHS):
                    snirf_channel_idxs = channel_idxs[moment][wavelength]["snirf_channel_idxs"]
                    unique_channel_idxs = channel_idxs[moment][wavelength]["unique_channel_idxs"]
                    to_send[moment, unique_channel_idxs, wavelength] = sample[snirf_channel_idxs]

            yield to_send
