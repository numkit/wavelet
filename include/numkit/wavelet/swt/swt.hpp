/// @file swt.hpp
/// @ingroup group_wavelet
// toolboxes/wavelet/include/nk/wavelet/swt/swt.hpp
//
// Shift-invariant wavelet transforms — the stationary DWT (swt) and
// the maximal-overlap DWT (modwt), plus their inverses.

#pragma once

#include <memory_resource>
#include <numkit/value/value.hpp>
#include <string>
#include <utility>

namespace numkit::wavelet {

/// @addtogroup group_wavelet
/// @{


/// Stationary (à trous) discrete wavelet transform (`swc = swt(x, n, wname)`).
///
/// Computes `n` levels of the **non-decimated** wavelet transform.
/// Each level uses upsampled-by-2 filters instead of downsampling
/// the signal, so every band has length `N` (shift-invariant).
///
/// Output layout: `(n+1) × N` matrix.
///   - rows 1..n : detail coefficients @f$ D_j @f$ at levels 1..n
///     (level 1 = finest).
///   - row n+1   : approximation @f$ A_n @f$ at the coarsest level.
///
/// Boundary: periodic (the default for swt).
///
/// @param x      Input signal, length must divide @f$ 2^n @f$.
/// @param n      Decomposition depth (≥ 1).
/// @param wname  Wavelet name.
/// @param mr     Memory resource (nullptr → process default).
/// @return       (n+1) × N matrix of coefficients.
/// @throws       Error if `length(x)` is not a multiple of @f$ 2^n @f$.
///
/// @see iswt, modwt
Value swt(const Value &x, int n, const std::string &wname,
          std::pmr::memory_resource *mr = nullptr);

/// Two-output stationary DWT (`[swa, swd] = swt(x, n, wname)`).
///
/// Same transform, but returns the per-level approximations and details
/// as separate n × N matrices: `swa(k, :)` is the approximation at
/// level k (level 1 = finest), `swd(k, :)` the matching detail. The
/// stacked @ref swt output is `[swd; swa(n, :)]`.
///
/// @param x      Input signal, length must divide @f$ 2^n @f$.
/// @param n      Decomposition depth (≥ 1).
/// @param wname  Wavelet name.
/// @param mr     Memory resource (nullptr → process default).
/// @return       (SWA, SWD) pair of n × N matrices.
std::pair<Value, Value> swtAD(const Value &x, int n, const std::string &wname,
                              std::pmr::memory_resource *mr = nullptr);

/// Inverse stationary DWT (`x = iswt(swc, wname)`).
///
/// Recovers the original signal from the `swc` matrix produced by
/// @ref swt by inverting the cascade level-by-level and averaging
/// shift variants at each step.
///
/// @param swc    (n+1) × N coefficient matrix from @ref swt.
/// @param wname  Wavelet name (must match the forward transform).
/// @param mr     Memory resource (nullptr → process default).
/// @return       Reconstructed row vector of length N.
///
/// @see swt
Value iswt(const Value &swc, const std::string &wname,
           std::pmr::memory_resource *mr = nullptr);

/// Two-array inverse stationary DWT (`x = iswt(swa, swd, wname)`).
///
/// Same reconstruction with the approximation and detail coefficients
/// supplied as separate matrices (the reference's SWA / SWD form; only
/// the LAST row of `swa` participates). The result class follows `swa`.
///
/// @param swa    Approximation coefficients (one row used).
/// @param swd    n × N detail matrix (level 1 = finest, row n = coarsest).
/// @param wname  Wavelet name (must match the forward transform).
/// @param mr     Memory resource (nullptr → process default).
/// @return       Reconstructed row vector of length N.
///
/// @see swt
Value iswt(const Value &swa, const Value &swd, const std::string &wname,
           std::pmr::memory_resource *mr = nullptr);

/// Maximal-overlap discrete wavelet transform (`swc = modwt(x, n, wname)`).
///
/// Like @ref swt but the filters are scaled by @f$ 1/\sqrt{2} @f$ per
/// level so the transform is energy-preserving (Parseval's identity
/// holds), and the signal length does **not** need to divide
/// @f$ 2^n @f$ (a true shift-invariant transform).
///
/// Output layout: `(n+1) × N` matrix.
///   - rows 1..n : wavelet coefficients @f$ W_j @f$ at levels 1..n.
///   - row n+1   : scaling coefficients @f$ V_n @f$ (final approximation).
///
/// Periodic boundary.
///
/// @param x      Input signal.
/// @param n      Decomposition depth.
/// @param wname  Wavelet name.
/// @param mr     Memory resource (nullptr → process default).
/// @return       (n+1) × N coefficient matrix.
///
/// @see imodwt, swt
Value modwt(const Value &x, int n, const std::string &wname,
            std::pmr::memory_resource *mr = nullptr);

/// Inverse MODWT (`x = imodwt(swc, wname)`).
///
/// Exact left-inverse of @ref modwt (`imodwt(modwt(x)) == x` up to
/// rounding).
///
/// @param swc    (n+1) × N coefficient matrix from @ref modwt.
/// @param wname  Wavelet name.
/// @param mr     Memory resource (nullptr → process default).
/// @return       Reconstructed row vector of length N.
///
/// @see modwt
Value imodwt(const Value &swc, const std::string &wname,
             std::pmr::memory_resource *mr = nullptr);


/// @}
} // namespace numkit::wavelet
