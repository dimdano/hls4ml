"""AXI-stream PYNQ driver for hls4ml VivadoAccelerator overlays.

Talks to a single AXI DMA IP (`hier_0/axi_dma_0`): sendchannel (PS -> PL, model
input) and recvchannel (PL -> PS, model output).
"""

from datetime import datetime

import numpy as np
from pynq import Overlay, allocate


class NeuralNetworkOverlay(Overlay):
    def __init__(
        self, bitfile_name, x_shape, y_shape, dtype=np.float32, dtbo=None, download=True, ignore_version=False, device=None
    ):
        super().__init__(bitfile_name, dtbo=None, download=True, ignore_version=False, device=None)
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dtype = dtype

    def _print_dt(self, timea, timeb, N):
        dt = timeb - timea
        dts = dt.seconds + dt.microseconds * 10**-6
        rate = N / dts
        print(f"Classified {N} samples in {dts} seconds ({rate} inferences / s)")
        return dts, rate

    def predict(self, X, debug=False, profile=False, encode=None, decode=None):
        """Run inference on the FPGA.

        X: ndarray matching x_shape (a single sample or a batch).
        encode/decode: optional float<->fixed-point conversion, only needed if the
            accelerator's AXI-stream type isn't plain float (ours is, so unused here).
        profile: if True, also returns (elapsed_seconds, inferences_per_second).
        """
        if profile:
            timea = datetime.now()
        if encode is not None:
            X = encode(X)

        # Buffers allocated fresh per call and released on exit, rather than reused
        # across calls: reusing the same buffer object across many DMA transfers was
        # found to eventually corrupt the DMA's internal descriptor/address tracking
        # on this board/pynq version. Matches a known-working reference driver
        # (PYNQ_files/MNIST/PYNQ_MNIST.ipynb).
        with (
            allocate(shape=self.x_shape, dtype=self.dtype) as input_buffer,
            allocate(shape=self.y_shape, dtype=self.dtype) as output_buffer,
        ):
            input_buffer[:] = X
            self.hier_0.axi_dma_0.sendchannel.transfer(input_buffer)
            self.hier_0.axi_dma_0.recvchannel.transfer(output_buffer)
            if debug:
                print("Transfer OK")
            self.hier_0.axi_dma_0.sendchannel.wait()
            if debug:
                print("Send OK")
            self.hier_0.axi_dma_0.recvchannel.wait()
            if debug:
                print("Receive OK")
            result = output_buffer.copy()
            input_buffer.flush()

        if decode is not None:
            result = decode(result)

        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            return result, dts, rate
        return result
