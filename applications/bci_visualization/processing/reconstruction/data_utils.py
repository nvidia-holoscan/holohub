import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .types import ChannelHeadsetMapping

logger = logging.getLogger(__name__)


def get_channel_mask(
    optode_order: List[Tuple[int, int, int, int]],
    headset_mapping: ChannelHeadsetMapping,
    mask_size: int,
) -> NDArray[np.bool_]:
    """Get channel mask for Jacobian slicing based on mapping.

    Parameters
    ----------
    optode_order : List[Tuple[int, int, int, int]]
        Optode tuples ordered the same as the moments channel axis.
    headset_mapping : ChannelHeadsetMapping
        Mapping of source/detector optodes to jacobian indices.
    mask_size : int
        Total feature-channels in the mega jacobian

    Returns
    -------
    NDArray[np.bool_]
        A boolean mask identifying which headset channels were populated.
    """
    channel_mask = np.zeros(mask_size, dtype=bool)
    for _channel_idx, (src_module, src, det_module, det) in enumerate(optode_order):
        try:
            srcs = headset_mapping[str(src_module)]
            detectors = srcs[str(src)][str(det_module)]
            jacobian_index = detectors[str(det)]
        except KeyError:
            raise ValueError(
                f"Channel without jacobian mapping (src_module={src_module}, src={src}, det_module={det_module}, det={det})",
            )

        channel_mask[jacobian_index] = True

    assert np.any(channel_mask)
    return channel_mask
