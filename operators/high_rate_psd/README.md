<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->
# High Rate PSD Operator

## Overview

A thin wrapper over the MatX [`abs2()`](https://nvidia.github.io/MatX/api/math/misc/abs2.html)
executor.

## Description

The high rate PSD operator...
- takes in a tensor of complex float data,
- performs a squared absolute value operation on the tensor: real(t)^2 + imag(t)^2,
- divides by the number of input elements
- emits the resultant tensor

## Requirements

- [MatX](https://github.com/NVIDIA/MatX)

## Example Usage

For an example of how to use this operator, see the
[`psd_pipeline_sim`](../../applications/psd_pipeline_sim) application.

## Configuration

The operator only takes one parameter: `burst_size`. This
is the number of samples to process on each invocation of `compute()`.
