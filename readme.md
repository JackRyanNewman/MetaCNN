## Overview

This project is a configurable CNN system built through a custom meta-compiler written in C (90%) and Python (10%). It generates and trains CNN architectures with configurable parameters for threading, precision, and SIMD vectorization. Python meta-compiles the C source (`CNN.c`), dynamically generating macros and de-virtualized code, and selects the most efficient computational strategies for each layer. The system achieves up to 4× faster performance than optimized NumPy implementations for data normalization, randomization, and loading, and introduces a heavily optimized homogeneous minibatching method that could be extended into PyTorch’s toolchain.

Beyond performance engineering, this project serves as a research platform for understanding how **SIMD vectorization, multithreading, cache hierarchies, and compiler optimizations** interact in modern CPUs. It explores how compilers manage memory alignment, prefetching, and instruction-level parallelism—and where manual optimization can outperform automatic heuristics. As the project evolves, new research questions naturally emerge and are documented along the way. All notes, experiments, and references are compiled in the **Research and Notes** folder, which also contains a PDF summarizing the findings and progress so far.

The research investigates:

- What optimizations are performed automatically by compilers versus what requires manual control.
- How cache access patterns, traversal direction, and memory layout influence prefetch efficiency.
- The interaction between SIMD and threading in relation to hardware prefetchers and cache latency.
- The potential use of experimental numeric formats (e.g., brainfloats) for CNN computation.

By combining compiler-level experimentation with practical CNN optimization, this work aims to clarify how low-level systems behavior impacts neural network performance, and how to deliberately exploit it for faster, more efficient computation.
