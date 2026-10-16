# numkit/wavelet

[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)

The **Wavelet Toolbox** for **[NumKit](https://github.com/numkit/numkit)**.

High-performance discrete, stationary, and continuous wavelet transforms, filter banks, and denoising algorithms with MATLAB R2025b parity.

---

## Features

- **Wavelet Filters (`filt`)**: `wfilters`, `qmf`, `wrev`, `dbwavf`, `coifwavf`, `symwavf`, `orthfilt` (Daubechies, Symlets, Coiflets, Biorthogonal, Reverse Biorthogonal, Haar, Discrete Meyer).
- **1-D Discrete Wavelet Transform (`dwt`)**: `dwt`, `idwt`, `wavedec`, `waverec`, `appcoef`, `detcoef`, `wrcoef`, `upcoef`, `wenergy`, `haart`, `ihaart`, `dyaddown`, `dyadup`, `wmaxlev`, `wkeep`, `wextend`.
- **2-D Discrete Wavelet Transform (`dwt2`)**: `dwt2`, `idwt2`, `wavedec2`, `waverec2`, `appcoef2`, `detcoef2`.
- **Stationary & Maximal Overlap Wavelet Transforms (`swt`)**: `swt`, `iswt`, `modwt`, `imodwt`.
- **Denoising & Compression (`denoise`)**: `wdenoise`, `wthresh`, `wentropy`, `ddencmp`, `wnoisest`.
- **Wavelet Shapes (`shape`)**: `mexihat`, `morlet`, `meyeraux`, `shanwavf`, `cmorwavf`, `fbspwavf`, `gauswavf`, `cgauwavf`.

---

## Building as a Dynamic Module

```bash
cmake -B build -S .
cmake --build build --config Release
```

Produces `numkit_wavelet.dll` (Windows) or `libnumkit_wavelet.so` (Linux/macOS).

---

## Usage in NumKit

Install via the NumKit package manager:

```bash
nk pkg install wavelet
```

Or load dynamically in a script:

```matlab
% Loaded automatically on first call to any wavelet function:
[a, d] = haart([1 2 3 4 5 6 7 8], 1);
disp(a);
```

---

## License

This software is released under the **[BSD Zero Clause License (0BSD)](LICENSE)**.
