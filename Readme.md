# Vitis project during the Xilinx XACC 2021 school

This project contains some C++ HLS code to accelerate a subpart of the
SIFT algorithm on Xilinx FPGA with the Vitis software.
See [the original paper of SIFT](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)
(published at [ICCV 1999 by David Lowe](https://dblp.org/rec/conf/iccv/Lowe99))
and [a complete implementation of it](https://github.com/robertwgh/ezSIFT).

This project derives from the [Difference of Gaussian HLS kernel](https://github.com/Xilinx/Vitis_Libraries/tree/2020.1/vision/L3/examples/gaussiandifference)
of the [Vitis vision library](https://xilinx.github.io/Vitis_Libraries/vision/2020.1/index.html). Host code uses OpenCL. Kernel code uses HLS DATAFLOW pragma and set the stream depths.


