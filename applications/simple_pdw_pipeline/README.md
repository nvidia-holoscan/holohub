# Basic Pulse Description Word (PDW) Generator

This is a Holoscan pipeline that shows the possibility of using Holoscan as a
Pulse Description Word (PDW) generator. This is a process that takes in IQ
samples (signals represented using time-series complex numbers) and picks out
peaks in the signal that may be transmissions from another source. These PDW
processors are used to see what is transmitting in your area, be they radio
towers or radars.

siggen.c a signal generator written in C that will transmit
the input to this pipeline.

## BasicNetworkOpRx

This uses the Basic Network Operator to read udp packets this operator is
documented elsewhere.

## PacketToTensorOp

This converts the bytes from the Basic Network Operator into the packets used
in the rest of the pipeline. The format of the incoming packets is a 16-bit id
followed by 8192 IQ samples each sample has the following format:
16 bits (I)
16 bits (Q)

## FFTOp

Does what it says on the tin. Takes an FFT of the input data. Also shifts data
so that 0 Hz is centered.

## ThresholdingOp

Detects samples over a threshold and then packetizes the runs of samples that
are above the threshold as a "pulse".

## PulseDescriptiorOp

Takes simple statistics of input pulses. This is where I am most excited for
future work, but that is not the point of this particular project.

## PulsePrinterOp

Prints the pulse to screen. Also optionally sends packets to a BasicNetworkOpTx.
The transmitted network packets have the following format:
Each of the following fields are 16bit unsigned integers
  id
  low bin
  high bin
  zero bin
  sum power
  max amplitude
  average amplitude
