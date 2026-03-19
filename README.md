ApexUltimateSOTA

High-performance fused GEMV + GELU implementation optimized for AVX-512 CPUs (Zen 4 / Ice Lake / Sapphire Rapids).

This library provides an ultra-optimized matrix-vector multiplication (GEMV) fused with the fast GELU activation function, using advanced techniques such as Chebyshev polynomial approximation, AVX-512 intrinsics, software prefetching, and non-temporal stores.

Features

Fused GEMV + GELU: Performs the matrix-vector product and GELU activation in a single pass to reduce memory traffic.

High-precision tanh approximation: Uses a 9th-order Chebyshev polynomial with Horner’s scheme for maximum numerical stability.

AVX-512 optimized: Fully utilizes 512-bit SIMD registers and multiple accumulators to saturate FMA execution ports.

Software prefetching: Improves cache usage for large matrices.

Non-temporal stores: Avoids polluting L1/L2 caches when writing output.

Tail handling: Masked operations ensure correctness when the number of rows is not a multiple of 16.

Requirements

CPU with AVX-512 support (Zen 4, Ice Lake, or Sapphire Rapids recommended).

C++17 or newer.

Compiler supporting AVX-512 intrinsics (GCC, Clang, or ICC).
