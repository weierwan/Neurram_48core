# NeuRRAM Compute-In-Memory Chip Testing Software

This repository contains software tool-chain for testing and deploying AI tasks on the NeuRRAM chip -- an RRAM based 48-core Compute-In-Memory chip for edge AI inference. 

The chip is tested on a [circuit board](board_design/) that houses an Opal Kelly FPGA module. The FPGA acts as a bridge to communicate between a host PC and the NeuRRAM chip.

[HDL/](HDL/) contains Verilog codes on the FPGA. They generate waveforms for NeuRRAM chip controls and data communications, and for several peripheral ICs on the test board.

[python/](python/) contains a Python based software tool-chain and APIs at various levels to interface with the chip and to map and execute AI workloads on the chip. The low level APIs provide access to basic operations of chip modules such as RRAM read and write and neuron analog-to-digital conversion; the middle level APIs include essential operations for implementing DNN layers such as (multi-core parallel) matrix-vector multiplications with configurable bit-precision and RRAM write-verify programming; the high level APIs integrate middle-level modules to implement weight mappers and run-times for complete DNN layers, such as convolutional and fully-connected layers.

[cifar10_resnet/](cifar10_resnet/), [mnist_cnn/](mnist_cnn/), [lstm/](lstm/), and [rbm/](rbm/) contain training and on-chip deployment codes and trained checkpoints for 4 different AI tasks. The deployment is implemented in Jupyter Notebooks (with file name ending with chip_inference.ipynb).
