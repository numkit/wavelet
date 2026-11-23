/// @file dwt.hpp
/// @ingroup group_wavelet
// toolboxes/wavelet/include/numkit/wavelet/dwt/dwt.hpp

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

namespace numkit::wavelet {

/// @addtogroup group_wavelet
/// @{


/// Single-level discrete wavelet transform (`[cA, cD] = dwt(x, wname)`).
///
/// Decomposes the 1-D signal `x` into an approximation band `cA`
/// (lowpass) and a detail band `cD` (highpass), each downsampled by 2.
/// The extension mode is fixed at `'sym'` (whole-point symmetric
/// reflection).
///
/// Output length per band:
/// @f$ \lfloor (N + L_f - 1)/2 \rfloor @f$ where @f$ L_f @f$ is the
/// filter length.
///
/// @param x      Input signal (vector).
/// @param wname  Wavelet name — `"haar"`, `"db1".."db10"`,
///               `"sym2".."sym10"`, `"coif1".."coif5"`.
/// @param mr     Memory resource (nullptr → process default).
/// @return       `(cA, cD)`; bind via `auto [cA, cD] = dwt(x, wname);`.
/// @throws       Error on unknown wavelet name.
///
/// @code
/// auto [cA, cD] = dwt(signal, "db4");
/// @endcode
///
/// @see idwt, wavedec, wfilters
keel::Pair<Value, Value>
dwt(const Value &x, keel::StringRef wname,
    keel::Allocator *mr = nullptr);

/// Internal entry: dwt with explicit analysis filters.
///
/// Mirrors @ref idwt_with_filters_pub for the forward transform — lets
/// the `dwt(x, Lo_D, Hi_D)` custom-filter form be invoked without
/// re-resolving filters by name. Public so the register TU (and other
/// toolboxes/wavelet TUs) can call it; not commonly needed by end users.
///
/// @param x    Input signal (vector).
/// @param Lo_D Analysis lowpass filter coefficients.
/// @param Hi_D Analysis highpass filter coefficients.
/// @param mr   Memory resource (nullptr → process default).
/// @return     `(cA, cD)` approximation / detail bands.
///
/// @see dwt, idwt_with_filters_pub
keel::Pair<Value, Value>
dwt_with_filters_pub(const Value &x,
                     keel::ArrayRef<keel::f64> Lo_D,
                     keel::ArrayRef<keel::f64> Hi_D,
                     keel::Allocator *mr = nullptr);

/// Inverse single-level discrete wavelet transform.
///
/// Reconstructs the original signal from an approximation / detail
/// pair. `len` (when ≥ 0) crops the result to that length; pass -1
/// to use the natural reconstruction length
/// `2·length(cA) - L_f + 2` (natural length for `'sym'` extension mode).
///
/// @param cA     Approximation coefficients (from @ref dwt).
/// @param cD     Detail coefficients.
/// @param wname  Wavelet name (must match the one used in @ref dwt).
/// @param len    Output length (-1 for the natural default).
/// @param mr     Memory resource (nullptr → process default).
/// @return       Reconstructed row vector of length `len` (or natural).
///
/// @see dwt, idwt_with_filters_pub
Value idwt(const Value &cA, const Value &cD,
           keel::StringRef wname,
           keel::i64 len /* = -1 */,
           keel::Allocator *mr = nullptr);

/// Internal entry: idwt with explicit synthesis filters.
///
/// Used by @ref appcoef to support the `(c, l, Lo_R, Hi_R)`
/// custom-filter form without re-resolving filters by name on every
/// cascade step. Public so other toolboxes/wavelet TUs can call it; not
/// commonly needed by end users.
///
/// @param cA   Approximation coefficients.
/// @param cD   Detail coefficients.
/// @param Lo_R Synthesis lowpass filter coefficients.
/// @param Hi_R Synthesis highpass filter coefficients.
/// @param len  Output length (-1 for natural).
/// @param mr   Memory resource (nullptr → process default).
/// @return          Reconstructed row vector.
///
/// @see idwt, appcoef
Value idwt_with_filters_pub(const Value &cA, const Value &cD,
                            keel::ArrayRef<keel::f64> Lo_R,
                            keel::ArrayRef<keel::f64> Hi_R,
                            keel::i64 len,
                            keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::wavelet
