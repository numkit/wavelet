// toolboxes/wavelet/src/dwt/ihaart.cpp
//
// 1-D inverse Haar discrete wavelet transform — ihaart.
//
// MATLAB R2025b semantics (verified end-to-end via probe scripts):
//
//   xrec = ihaart(a, d[, level[, integerflag]])
//
//   * `a` — approximation column / matrix from haart.
//   * `d` — detail. Plain matrix when haart was called with level=1,
//     a length-Nlevels cell array otherwise; d{1} is the finest scale.
//   * `level` (default 0) — non-negative integer in [0, Nlevels). When
//     level=K the detail bands d{1..K} are zeroed BEFORE reconstruction
//     (so xrec is still full-length but with the finest K scales
//     suppressed). level==0 means lossless reconstruction.
//   * `integerflag` ∈ {'noninteger' (default), 'integer'} chooses the
//     orthogonal pair (factor 1/√2) or the lifting integer Haar inverse:
//
//       noninteger:  x[2k]   = (a[k] - d[k]) / √2
//                    x[2k+1] = (a[k] + d[k]) / √2
//       integer:     x[2k]   = a[k] - floor(d[k] / 2)
//                    x[2k+1] = x[2k] + d[k]
//
//   * `a` may be complex; `d` MUST be real and finite — a non-real or
//     non-finite `d` is rejected with an error.
//   * Empty `a` errors. Negative level errors. level >= Nlevels errors.
//   * Output orientation: vector-shaped a returns column; matrix
//     returns matrix (columns reconstructed independently).

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/dwt/ihaart.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The ihaart builtin
// (CallContext wrapper) lives in dwt/haar_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::wavelet {

namespace {

void ihaar_step_real(const keel::f64 *a, const keel::f64 *d, keel::usize H,
                     keel::f64 *out, bool integer)
{
    keel::wavelet::ihaar_step(keel::ArrayRef<keel::f64>(a, H),
                              keel::ArrayRef<keel::f64>(d, H),
                              keel::MutArrayRef<keel::f64>(out, 2 * H), integer);
}

void ihaar_step_cplx(const keel::c64 *a, const keel::f64 *d, keel::usize H, keel::c64 *out)
{
    keel::wavelet::ihaar_step_c64(keel::ArrayRef<keel::c64>(a, H),
                                  keel::ArrayRef<keel::f64>(d, H),
                                  keel::MutArrayRef<keel::c64>(out, 2 * H));
}

// Pull a single detail level (length H × ncols) out of the user-supplied
// `d`. If `d` is a cell, take cell at `idx` (0-based, where 0 is finest).
// If `d` is a plain matrix the only valid idx is 0.
//
// Returns:
//   - rows / cols of the level
//   - a vector<double> with the level's values (column-major); if
//     `zero_out` is true the returned vector is zero-filled (used to
//     suppress the finest K bands when `level > 0`).
struct DetailLevel {
    keel::Array<keel::f64> data;
    keel::usize rows = 0;
    keel::usize cols = 0;
};

DetailLevel grab_level(const Value &d, bool isCell, keel::usize idx,
                       keel::usize expectedRows, keel::usize expectedCols,
                       bool zero_out)
{
    DetailLevel L;
    const Value &m = isCell ? d.cellAt(idx) : d;
    if (m.numel() == 0)
        throw Error("ihaart: detail level is empty",
                    0, 0, "ihaart", "", "numkit:ihaart:detail");
    if (m.isComplex())
        throw Error("ihaart: expected input number 2, D, to be real",
                    0, 0, "ihaart", "", "numkit:ihaart:complexD");

    // Detail shape is read directly (no row-↔-column coercion). For a
    // vector-input haart this is rows×1; for a matrix-input haart at
    // a deep level it can legitimately be 1×ncols.
    const keel::usize rows = m.dims().rows();
    const keel::usize cols = m.dims().cols();

    if (rows != expectedRows || cols != expectedCols)
        throw Error("ihaart: detail dimensions are inconsistent with the "
                    "approximation",
                    0, 0, "ihaart", "", "numkit:ihaart:dimMismatch");

    const keel::usize mn = rows * cols;

    L.rows = rows;
    L.cols = cols;
    L.data.resize(rows * cols);
    if (zero_out) {
        // already zero
        return L;
    }
    for (keel::usize i = 0; i < mn; ++i) L.data[i] = m.elemAsDouble(i);
    return L;
}

keel::usize ihaart_argcheck(const Value &aV, const Value &dV, keel::i32 level, bool integer)
{
    if (aV.numel() == 0)
        throw Error("ihaart: expected input number 1, A, to be nonempty",
                    0, 0, "ihaart", "", "numkit:ihaart:emptyA");

    // Probed R2025b class contract: a and d are double/single; a may be
    // complex (complex result), d must be real; logical / integer / char
    // reject on either side; the result class follows a alone
    // (double a + single d reconstructs in double).
    internal::requireWaveClass(aV, "ihaart", "A", 1,
                               internal::WvClasses::FloatOnly,
                               /*allowComplex=*/true);
    const bool isCell = dV.isCell();
    if (isCell) {
        for (keel::usize ci = 0; ci < dV.numel(); ++ci)
            internal::requireWaveClass(dV.cellAt(ci), "ihaart", "D", 2,
                                       internal::WvClasses::FloatOnly);
    } else {
        internal::requireWaveClass(dV, "ihaart", "D", 2,
                                   internal::WvClasses::FloatOnly);
    }

    // Determine cell-vs-plain detail and Nlevels.
    keel::usize Nlevels = 0;
    if (isCell) {
        const keel::usize nc = dV.dims().rows() * dV.dims().cols();
        if (nc == 0)
            throw Error("ihaart: detail cell array is empty",
                        0, 0, "ihaart", "", "numkit:ihaart:emptyD");
        Nlevels = nc;
    } else {
        if (dV.numel() == 0)
            throw Error("ihaart: detail array is empty",
                        0, 0, "ihaart", "", "numkit:ihaart:emptyD");
        if (dV.isComplex())
            throw Error("ihaart: expected input number 2, D, to be real",
                        0, 0, "ihaart", "", "numkit:ihaart:complexD");
        Nlevels = 1;
    }

    if (level < 0)
        throw Error("ihaart: expected input number 3, LEVEL, to be a scalar "
                    "with value >= 0",
                    0, 0, "ihaart", "", "numkit:ihaart:level");
    if (level >= keel::as<keel::i32>(Nlevels))
        throw Error(keel::format("ihaart: expected input number 3, LEVEL, to be a scalar "
                                 "with value < {}", Nlevels),
                    0, 0, "ihaart", "", "numkit:ihaart:level");
    if (integer && aV.isComplex())
        throw Error("ihaart: integer mode is not supported for complex data",
                    0, 0, "ihaart", "", "numkit:ihaart:flag");
    return Nlevels;
}

} // anonymous

// ── Public C++ API (see dwt/ihaart.hpp) ───────────────────────────────

Value ihaart(const Value &aV, const Value &dV, keel::i32 level, bool integer,
             keel::Allocator *mr)
{
    const keel::usize Nlevels = ihaart_argcheck(aV, dV, level, integer);
    const bool isCell = dV.isCell();

    // `a` is always shaped to match the detail dimensions (haart emits
    // columns for vector inputs), so the matrix shape is read directly.
    const keel::usize aRows = aV.dims().rows();
    const keel::usize aCols = aV.dims().cols();
    const keel::usize aLen  = aRows;
    const keel::usize ncols = aCols;

    const bool aIsComplex = aV.isComplex();
    const ValueType vt = aIsComplex ? ValueType::COMPLEX : ValueType::DOUBLE;

    // Working buffer: starts as `a`, doubles each level (column-major).
    keel::Array<keel::f64>  curR(mr);
    keel::Array<keel::c64> curC(mr);
    if (aIsComplex) {
        curC.resize(aLen * ncols);
        for (keel::usize c = 0; c < ncols; ++c)
            for (keel::usize r = 0; r < aLen; ++r)
                curC[c * aLen + r] = aV.complexElem(c * aRows + r);
    } else {
        curR.resize(aLen * ncols);
        for (keel::usize c = 0; c < ncols; ++c)
            for (keel::usize r = 0; r < aLen; ++r)
                curR[c * aLen + r] = aV.elemAsDouble(c * aRows + r);
    }
    keel::usize curLen = aLen;

    // Reconstruct coarsest -> finest. For cells d{1} is finest, so
    // iterating L = Nlevels down to 1 uses detail at index (L-1).
    for (keel::i32 L = keel::as<keel::i32>(Nlevels); L >= 1; --L) {
        const keel::usize idx = keel::as<keel::usize>(L - 1);     // 0..Nlevels-1
        const bool zero = (idx < keel::as<keel::usize>(level));
        DetailLevel det = grab_level(dV, isCell, idx, curLen, ncols, zero);

        const keel::usize outLen = curLen * 2;
        keel::Array<keel::f64>  newR(mr);
        keel::Array<keel::c64> newC(mr);
        if (aIsComplex) {
            newC.resize(outLen * ncols);
            for (keel::usize c = 0; c < ncols; ++c)
                ihaar_step_cplx(curC.data() + c * curLen,
                                det.data.data() + c * curLen, curLen,
                                newC.data() + c * outLen);
            curC = keel::move(newC);
        } else {
            newR.resize(outLen * ncols);
            for (keel::usize c = 0; c < ncols; ++c)
                ihaar_step_real(curR.data() + c * curLen,
                                det.data.data() + c * curLen, curLen,
                                newR.data() + c * outLen, integer);
            curR = keel::move(newR);
        }
        curLen = outLen;
    }

    // Pack the result. Vector input -> column output; matrix -> matrix.
    // Single a reconstructs in single (the class follows a, probed).
    const bool aSingle = aV.type() == ValueType::SINGLE && !aIsComplex;
    Value out = Value::matrix(curLen, ncols,
                              aSingle ? ValueType::SINGLE : vt, mr);
    if (aIsComplex) {
        keel::c64 *p = out.complexDataMut();
        keel::copy_n(p, curC.data(), curC.size());
    } else if (aSingle) {
        keel::f32 *p = out.singleDataMut();
        for (keel::usize i = 0; i < curR.size(); ++i)
            p[i] = keel::as<keel::f32>(curR[i]);
    } else {
        keel::f64 *p = out.doubleDataMut();
        keel::copy_n(p, curR.data(), curR.size());
    }
    return out;
}

} // namespace numkit::wavelet
