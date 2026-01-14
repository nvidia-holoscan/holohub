"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

import logging
from queue import Empty, Full, Queue
from threading import Thread, Event as ThreadingEvent
from typing import Iterator, List

import numpy as np
from kernel.sdk import MomentNumber, SdkClient, Wavelength
from kernel.sdk.socket import (
    Event,
    check_faulted_on_command_failure,
    requires_flow_booted,
    requires_flow_connected,
)

from .base_nirs import BaseNirsStream, ChannelInfo

logger = logging.getLogger(__name__)


class KernelSDKReceiver(SdkClient):
    """
    A class to receive data from the Flow device using the Kernel SDK.
    """
    _receiver_stop_event = ThreadingEvent()

    @property
    @check_faulted_on_command_failure
    @requires_flow_connected
    @requires_flow_booted
    def iter(self) -> Iterator[np.ndarray]:
        """
        Get an infinite iterator over the flow device's data.

        subscribes to nirs data and yields samples
        """
        data_queue: Queue[np.ndarray] = Queue(5)
        moment_ids = [MomentNumber.Zeroth, MomentNumber.First, MomentNumber.Second]
        wavelengths = [Wavelength.Red, Wavelength.IR]

        # need to yield  (num_modules, num_sources, num_modules, num_detectors, num_moments, num_wavelengths) shaped data into data_queue

        def _multi_moments_callback(data_dict: dict):
            logger.debug("NIRS data callback triggered")

            moment_slices = []
            for moment_id in moment_ids:
                wavelength_slices = []
                for wavelength in wavelengths:
                    urn = self.MOMENTS_FIELD_URN_TEMPLATE.format(
                        moment_id=moment_id.value, wavelength=wavelength.value
                    )
                    # (num_samples, num_modules, num_sources, num_modules, num_detectors)
                    field_data = data_dict.get(urn)
                    if field_data is None:
                        logger.warning(
                            f"Missing data for {urn} among {list(data_dict.keys())}, skipping callback."
                        )
                        return
                    # wavelength_slices is a list of (num_samples, num_modules, num_sources, num_modules, num_detectors)
                    wavelength_slices.append(field_data)
                # moment_slices is a list of (num_samples, num_modules, num_sources, num_modules, num_detectors, num_wavelengths)
                moment_slices.append(np.stack(wavelength_slices, axis=-1))
            # stacked_data is (num_samples, num_modules, num_sources, num_modules, num_detectors, num_moments, num_wavelengths)
            stacked_data = np.stack(moment_slices, axis=-2)

            for sample in stacked_data:
                try:
                    data_queue.put_nowait(sample)
                except Full:
                    # this happens when the pipeline is not consuming fast enough
                    # mainly during cache/warmup phase
                    logger.debug("Data queue is full. Dropping sample.")

        # Subscribe to all moment_id and wavelength combinations
        urns = [
            self.MOMENTS_FIELD_URN_TEMPLATE.format(
                moment_id=moment_id.value, wavelength=wavelength.value
            )
            for moment_id in moment_ids
            for wavelength in wavelengths
        ]
        logger.info(f"Subscribing to NIRS data URNs: {urns}")
        nirs_future = self._sdk.new_event(
            Event(device="flow", field_urns=urns), _multi_moments_callback
        )

        try:
            while not self._receiver_stop_event.is_set():
                # Block until data is available in the queue
                logger.debug("Waiting for NIRS data...")
                yield data_queue.get()
                logger.debug("NIRS data received. queue size: %d", data_queue.qsize())
        finally:
            nirs_future.cancel()

    def stop(self) -> None:
        """
        Stop the receiver.
        """
        self._receiver_stop_event.set()
        super().stop()


class KernelSDKStream(BaseNirsStream):
    MAX_MODULE_COUNT = 48
    NUM_SOURCES = 3
    NUM_DETECTORS = 6

    def __init__(
        self,
        *,
        receiver_queue_size: int = 32,
    ) -> None:
        if receiver_queue_size <= 0:
            raise ValueError("receiver_queue_size must be positive")
        self._receiver_queue_size = int(receiver_queue_size)
        self._receiver_queue: Queue[np.ndarray] | None = None
        self._receiver_thread: Thread | None = None
        self._channels: ChannelInfo | None = None
        self._good_channels: List[int] = []

    def start(self) -> None:
        if self._receiver_thread is not None and self._receiver_thread.is_alive():
            raise RuntimeError("KernelSDKStream is already started")

        self._receiver = KernelSDKReceiver()
        if self._receiver_queue is None:
            self._receiver_queue = Queue(self._receiver_queue_size)
        self._receiver_thread = Thread(
            target=self._receiver_loop,
            name="KernelSDKReceiverThread",
            daemon=True,
        )
        self._receiver_thread.start()
        self._channels = self._build_all_channels()

    def stop(self) -> None:
        """Stop threads and clean up resources."""
        if self._receiver_queue is not None:
            # Signal thread to stop by setting queue to None
            queue = self._receiver_queue
            self._receiver_queue = None
            # Drain the queue to unblock the thread
            try:
                while True:
                    queue.get_nowait()
            except Empty:
                pass
        
        if self._receiver_thread is not None and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=2.0)
            self._receiver_thread = None

        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None

    def _build_all_channels(self) -> ChannelInfo:
        nirs_data: np.ndarray
        if self._receiver_queue is None:
            raise RuntimeError("KernelSDKStream.start() must be called before streaming NIRS data")

        try:
            # nirs_data is (num_modules, num_sources, num_modules, num_detectors, num_moments, num_wavelengths) = (48, 3, 48, 6, 3, 2)
            nirs_data = self._receiver_queue.get(timeout=1.0)
            logger.debug("Received new NIRS data frame for channel building")
        except Empty:
            raise RuntimeError(
                "Timeout waiting for NIRS data from Kernel SDK. "
                "Ensure the device is connected and streaming."
            )

        detector_module = []
        detector_number = []
        source_module = []
        source_number = []

        channel_idx = -1
        for source_module_id in range(self.MAX_MODULE_COUNT):
            for source_num in range(self.NUM_SOURCES):
                for detector_module_id in range(self.MAX_MODULE_COUNT):
                    for detector_num in range(self.NUM_DETECTORS):
                        channel_idx += 1

                        # if there are any nan values for this channel, skip it
                        channel_data = nirs_data[
                            source_module_id, source_num, detector_module_id, detector_num
                        ]
                        # if any are not finite, skip
                        if not np.isfinite(channel_data).all():
                            continue

                        self._good_channels.append(channel_idx)

                        detector_module.append(detector_module_id)
                        detector_number.append(detector_num)
                        source_module.append(source_module_id)
                        source_number.append(source_num)

        return ChannelInfo(
            source_module=np.array(source_module),
            source_number=np.array(source_number),
            detector_module=np.array(detector_module),
            detector_number=np.array(detector_number),
        )

    def _receiver_loop(self):
        """
        Dedicated thread that receives data from the KernelSDKReceiver and puts it into a queue.
        Prevents internal queue filling up when stream_nirs is not consuming fast enough.
        """
        if self._receiver_queue is None:
            return
        try:
            for frame in self._receiver.iter:
                queue = self._receiver_queue
                if queue is None:
                    break
                try:
                    queue.put_nowait(frame)
                except Full:
                    # this happens when the pipeline is not consuming fast enough
                    # mainly during cache/warmup phase
                    logger.debug("KernelSDKStream receiver queue full; dropping frame")
        except Exception:
            logger.exception("KernelSDKStream receiver loop crashed")

    def get_channels(self) -> ChannelInfo:
        if self._channels is None:
            raise RuntimeError("KernelSDKStream.start() must be called before getting channels")
        return self._channels

    def stream_nirs(self):
        if self._receiver_queue is None:
            raise RuntimeError("KernelSDKStream.start() must be called before streaming NIRS data")
        while True:
            try:
                # nirs_data is (num_modules, num_sources, num_modules, num_detectors, num_moments, num_wavelengths) = (48, 3, 48, 6, 3, 2)
                nirs_data = self._receiver_queue.get(timeout=1.0)
            except Empty:
                raise RuntimeError(
                    "Timeout waiting for NIRS data from Kernel SDK. "
                    "Ensure the device is connected and streaming."
                )

            logger.debug("Received new NIRS data frame")

            # flatten first 4 dimensions into one dimension
            nirs_data = nirs_data.reshape(-1, nirs_data.shape[4], nirs_data.shape[5])

            # channels, moments, wavelengths => moments, channels, wavelengths
            reshaped = nirs_data.transpose(1, 0, 2)
            logger.debug("NIRS data shape after reshaping: %s", reshaped.shape)
            reshaped = reshaped[:, self._good_channels, :]
            logger.debug("NIRS data shape after channel filtering: %s", reshaped.shape)

            yield reshaped
