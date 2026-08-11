# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

__all__ = ["OnlineVideoReasonerOp"]


def __getattr__(name):
    """Import the Holoscan operator only when it is requested."""
    if name == "OnlineVideoReasonerOp":
        from .online_video_reasoner import OnlineVideoReasonerOp

        return OnlineVideoReasonerOp
    raise AttributeError(name)
