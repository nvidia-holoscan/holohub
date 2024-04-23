# Velodyne Lidar Operator

## Overview

A Holoscan operator to convert packets from the Velodyne VLP-16 Lidar sensor
to a point cloud tensor format.

## Description

This operator receives packets from a Velodyne VLP-16 lidar and
processes them into a point cloud of fixed size in Cartesian space.

The operator performs the following steps:
1. Interpret a fixed-size UDP packet as a Velodyne VLP-16 lidar packet,
   which contains 12 data blocks (azimuths) and 32 spherical data points per block.
2. Transform the spherical data points into Cartesian coordinates (x, y, z)
   and add them to the output point cloud tensor, overwriting a previous cloud segment.
3. Output the point cloud tensor and update the tensor insertion pointer to prepare
   for the next incoming packet.

We recommend relying on HoloHub networking operators to receive Velodyne VLP-16 lidar packets
over UDP/IP and forward them to this operator.

## Requirements

Hardware requirements:
- Holoscan supported platform (x64 or NVIDIA IGX devkit);
- Velodyne VLP-16 Lidar sensor

## Example Usage

See the [HoloHub Lidar Sample Application](../../applications/velodyne_lidar_app) to get started.

## Acknowledgements

This operator was developed in part with support from the NVIDIA nvMap team and adapts portions
of the NVIDIA DeepMap SDK.
