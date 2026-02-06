"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""
from math import pi

FS = 4.76 # Streaming frequency of Kernel Flow in Hz

class EmaHighPass:
    """
    Exponential Moving Average (EMA) High-Pass Filter.
    """

    def __init__(self, fs=FS, fc=0.01):
        self.alpha = (2 * pi * fc) / (fs + 2 * pi * fc)
        self.ema = None

    def step(self, x):
        """
        Process one sample (or vector of channels).
        """
        if self.ema is None:
            self.ema = x
        else:
            self.ema = self.alpha * x + (1 - self.alpha) * self.ema
        return x - self.ema