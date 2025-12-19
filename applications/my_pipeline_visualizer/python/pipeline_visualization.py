#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pipeline Visualization Application - Python Implementation

This example showcases a Holoscan pipeline that:
1. Generates a synthetic sine wave signal with time-varying frequency (10-20 Hz)
2. Adds high-frequency modulation (300 Hz) to simulate measurement noise
3. Processes the resulting signal through a sink operator
4. Optionally streams data to a NATS server for external visualization

The application demonstrates tensor manipulation, operator chaining, and data logging
capabilities in the Holoscan SDK.
"""

import argparse
from pathlib import Path

import numpy as np
from holoscan import as_tensor
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec

from holohub.nats_logger import NatsLogger


class SourceOp(Operator):
    """Source operator that generates a sine wave signal.

    This operator produces a synthetic time-series signal consisting of a sine wave
    with a gradually increasing frequency (10-20 Hz). It outputs 3000 samples per compute cycle.
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        # Because frequency has bee ndefined here, it will get retained between graph executions
        self.frequency = 10.0  # Current frequency of the sine wave in Hz

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        """
        Generate a sine wave signal with time-varying frequency.
        NOTE: Generates 1 second fragments of the sine curve that are 3000 points long.
        Each time the fragment is executed, the frequency will be incremented by 0.1 Hz.
        """
        # Generate a sine wave signal with time-varying frequency
        samples = 3000  # Number of samples in the signal
        duration = 1.0  # Duration of signal in seconds
        sample_time = duration / samples
        omega = 2 * np.pi * self.frequency  # Angular frequency

        # Generate the sine wave
        t = np.arange(samples, dtype=np.float32) * sample_time
        wave = np.sin(omega * t)

        # Reshape to (samples, 1) to match C++ implementation
        wave = wave.reshape(samples, 1)

        # Gradually increase frequency from 10 to 20 Hz, then wrap back to 10 Hz
        self.frequency += 0.1
        if self.frequency > 20.0:
            self.frequency = 10.0

        # Emit the tensor to the output port
        op_output.emit(dict(wave=as_tensor(wave)), "out")

class SourceIsingOp(Operator):
    """Source operator that generates a 2D Ising model simulation.

    This operator simulates a 64x64 2D Ising model using the Metropolis-Hastings algorithm.
    Each site has a spin (+1 or -1), and spins interact with their nearest neighbors.
    The system evolves toward thermal equilibrium at a given temperature.
    
    Physics:
    - Hamiltonian: H = -J * sum(s_i * s_j) for nearest neighbors
    - We use J=1 (ferromagnetic coupling)
    - Temperature in units where k_B=1
    - Critical temperature T_c ≈ 2.269 (Onsager solution)
    """

    def __init__(self, fragment, *args, temperature=2.5, lattice_size=64, sweeps_per_frame=5, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.size = lattice_size  # Lattice size (configurable)
        self.temperature = temperature  # Temperature (try 1.5, 2.269, 3.0 for different phases)
        self.beta = 1.0 / self.temperature  # Inverse temperature
        
        # Initialize spins randomly (+1 or -1)
        # Use -1 and +1 instead of 0 and 1 for proper Ising model
        self.spins = np.random.choice([-1, 1], size=(self.size, self.size)).astype(np.float32)
        
        # Statistics
        self.iteration = 0
        self.sweeps_per_frame = sweeps_per_frame  # Number of Monte Carlo sweeps per output frame
        
        print(f"[Ising Model] Initialized {self.size}x{self.size} lattice at T={self.temperature:.3f} (T_c≈2.269)")

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        """Perform Monte Carlo updates on the Ising model and emit the spin configuration."""
        
        # Perform multiple Monte Carlo sweeps per frame for visible evolution
        for _ in range(self.sweeps_per_frame):
            self._monte_carlo_sweep()
        
        # Calculate some statistics
        magnetization = np.mean(self.spins)
        energy = self._calculate_energy()
        
        if self.iteration % 20 == 0:
            print(f"[Ising] Iteration {self.iteration}: M={magnetization:+.3f}, "
                  f"E/N={energy/(self.size*self.size):.3f}, T={self.temperature:.3f}")
        
        # Create output tensor (shape: 64x64x1 for compatibility with visualizer)
        # Normalize to [0, 1] range for visualization: (spin + 1) / 2
        spin_config = ((self.spins + 1.0) / 2.0).reshape(self.size, self.size, 1).astype(np.float32)
        
        self.iteration += 1
        
        # Optional: Vary temperature over time to see phase transition
        # Uncomment to cycle temperature and observe order/disorder transition
        # self.temperature = 2.269 + 0.5 * np.sin(self.iteration * 0.02)
        # self.beta = 1.0 / self.temperature
        
        # Emit the spin configuration
        op_output.emit(dict(spins=as_tensor(spin_config)), "out")
    
    def _monte_carlo_sweep(self):
        """Perform one Monte Carlo sweep using Metropolis algorithm."""
        # Number of flip attempts per sweep (attempt to flip each spin once on average)
        n_attempts = self.size * self.size
        
        for _ in range(n_attempts):
            # Select random site
            i = np.random.randint(0, self.size)
            j = np.random.randint(0, self.size)
            
            # Current spin value
            spin = self.spins[i, j]
            
            # Sum of nearest neighbor spins (periodic boundary conditions)
            neighbors_sum = (
                self.spins[(i+1) % self.size, j] +
                self.spins[(i-1) % self.size, j] +
                self.spins[i, (j+1) % self.size] +
                self.spins[i, (j-1) % self.size]
            )
            
            # Energy change if we flip this spin
            # For Ising model: ΔE = 2*J*s_i*Σs_j (we use J=1)
            delta_E = 2.0 * spin * neighbors_sum
            
            # Metropolis acceptance criterion
            # Accept if ΔE < 0 (energy decreases)
            # Or accept with probability exp(-β*ΔE) if ΔE > 0
            if delta_E < 0 or np.random.random() < np.exp(-self.beta * delta_E):
                self.spins[i, j] *= -1  # Flip the spin
    
    def _calculate_energy(self):
        """Calculate total energy of the system."""
        energy = 0.0
        for i in range(self.size):
            for j in range(self.size):
                spin = self.spins[i, j]
                # Count each pair only once
                neighbors = (
                    self.spins[(i+1) % self.size, j] +
                    self.spins[i, (j+1) % self.size]
                )
                energy -= spin * neighbors  # E = -J * sum(s_i * s_j), J=1
        return energy

class ModulateOp(Operator):
    """Modulation operator that adds high-frequency noise to the input signal.

    This operator receives a time-series signal and adds a 300 Hz sinusoidal modulation
    with small amplitude (0.05) to simulate measurement noise or signal perturbation.
    """

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        """Add high-frequency modulation to the input signal."""
        # Receive the input tensor from the source operator
        input_tensor = op_input.receive("in").get("wave")
        samples = input_tensor.shape[0]

        # Parameters for high-frequency modulation/noise
        frequency = 300.0  # Modulation frequency in Hz
        amplitude = 0.05  # Amplitude of the modulation
        duration = 1.0  # Duration of signal in seconds
        sample_time = duration / samples
        omega = 2 * np.pi * frequency  # Angular frequency

        # Add modulation to the input signal
        t = np.arange(samples, dtype=np.float32) * sample_time
        modulation = amplitude * np.sin(omega * t).reshape(samples, 1)
        modulated_signal = input_tensor + modulation

        # Emit the modulated signal to the output port
        op_output.emit(dict(modulated_signal=as_tensor(modulated_signal)), "out")


class SinkOp(Operator):
    """Sink operator that consumes the processed signal.

    This operator acts as the terminal node in the pipeline, receiving the modulated
    signal. In this example, it simply receives the data without additional processing,
    but could be extended to perform analysis or visualization.
    """

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        """Receive and process the modulated signal."""
        # Receive the modulated signal from the previous operator
        tensormap = op_input.receive("in")
        _tensor = next(iter(tensormap.values()))  # get first tensor regardless of name
        # Note: In this example, we simply receive the tensor. Additional processing
        # or visualization could be added here.


class TimeSeriesApp(Application):
    """Main application class for pipeline visualization demo.

    This application creates a pipeline that generates a time-varying sine wave,
    adds high-frequency modulation to it, and processes the result. The application
    can optionally log data to a NATS server for external monitoring and visualization.

    Pipeline flow: SourceOp -> ModulateOp -> SinkOp
    """

    def __init__(
        self,
        disable_logger,
        nats_url,
        subject_prefix,
        publish_rate,
        *args,
        **kwargs,
    ):
        """Initialize TimeSeriesApp.

        Args:
            disable_logger: Whether to disable the NATS logger
            nats_url: URL for the NATS server connection
            subject_prefix: Prefix for NATS subject names
            publish_rate: Rate at which to publish data to NATS (Hz)
        """
        super().__init__(*args, **kwargs)
        self.disable_logger = disable_logger
        self.nats_url = nats_url
        self.subject_prefix = subject_prefix
        self.publish_rate = publish_rate

    def compose(self):
        """Compose the application pipeline."""
        # Read application configuration
        app_config = self.kwargs("application")
        use_ising = app_config.get("use_ising", False) if app_config else False
        
        if not self.disable_logger:
            # Create and configure the NATS logger for data streaming
            nats_logger = NatsLogger(
                self,
                name="nats_logger",
                nats_url=self.nats_url,
                subject_prefix=self.subject_prefix,
                publish_rate=self.publish_rate,
                **self.kwargs("nats_logger"),
            )
            # Register the logger with the application
            self.add_data_logger(nats_logger)

        if use_ising:
            # Ising model pipeline: just source -> sink (no modulation needed)
            ising_config = app_config.get("ising", {}) if app_config else {}
            lattice_size = ising_config.get("lattice_size", 64)
            print(f"[Pipeline] Using Ising Model ({lattice_size}x{lattice_size} 2D lattice)")
            
            source_op = SourceIsingOp(
                self,
                # Update every 50ms (20 Hz)
                PeriodicCondition(self, name="periodic-condition", recess_period=0.05),
                name="ising_source",
                **ising_config,  # Pass temperature, lattice_size, sweeps_per_frame
            )
            sink_op = SinkOp(self, name="sink")
            
            # Connect directly: ising_source -> sink
            self.add_flow(source_op, sink_op, {("out", "in")})
        else:
            # Original sine wave pipeline: source -> modulate -> sink
            print("[Pipeline] Using Sine Wave Generator")
            source_op = SourceOp(
                self,
                # Limit the rate of the source operator
                PeriodicCondition(self, name="periodic-condition", recess_period=0.05),
                name="source",
            )
            modulate_op = ModulateOp(self, name="modulate")
            sink_op = SinkOp(self, name="sink")

            # Connect the operators: source -> modulate -> sink
            self.add_flow(source_op, modulate_op, {("out", "in")})
            self.add_flow(modulate_op, sink_op, {("out", "in")})


def main():
    """Main entry point for the pipeline visualization application.

    Parses command-line arguments, configures the application, and runs the pipeline.
    Supports options for NATS server configuration and data logging control.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Holoscan Pipeline Visualization Demo - Python Implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--disable_logger",
        action="store_true",
        help="Disable NATS logger",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-u",
        "--nats_url",
        type=str,
        default="nats://0.0.0.0:4222",
        help="NATS server URL",
    )
    parser.add_argument(
        "-p",
        "--subject_prefix",
        type=str,
        default="nats_demo",
        help="NATS subject prefix",
    )
    parser.add_argument(
        "-r",
        "--publish_rate",
        type=float,
        default=5.0,
        help="Publish rate (Hz)",
    )

    args = parser.parse_args()

    # Determine config file path
    if args.config:
        config_path = args.config
    else:
        # Use default config file in the same directory as this script
        config_path = Path(__file__).parent / "pipeline_visualization.yaml"

    # Create the application instance with parsed parameters
    app = TimeSeriesApp(
        disable_logger=args.disable_logger,
        nats_url=args.nats_url,
        subject_prefix=args.subject_prefix,
        publish_rate=args.publish_rate,
    )

    # Load configuration from YAML file
    if config_path and Path(config_path).exists():
        app.config(str(config_path))

    # Execute the application pipeline
    app.run()


if __name__ == "__main__":
    main()
