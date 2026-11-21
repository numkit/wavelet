// toolboxes/wavelet/src/dwt/haart.cpp
//
// 1-D Haar discrete wavelet transform — haart.
//
// MATLAB R2025b semantics (verified end-to-end via probe scripts):
//
//   [a, d] = haart(x[, level[, integerflag]])
//
//   * x is a vector (row or column) or a matrix; column orientation is
//     forced on the output regardless of input orientation. For matrix
//     input each column is processed independently and `rows` must be
//     even (and divisible by 2^level at each level).
//   * `level` (default = max k such that 2^k divides length(x))
//     positive integer; must satisfy level ≤ max level.
//   * `integerflag` ∈ {'noninteger' (default), 'integer'} selects the
//     orthogonal Haar pair (factor 1/√2) or the lifting integer Haar:
//
//       noninteger:   a[k] = (x[2k] + x[2k+1]) / √2
//                     d[k] = (x[2k+1] - x[2k]) / √2
//       integer:      d[k] = x[2k+1] - x[2k]
//                     a[k] = x[2k] + floor(d[k] / 2)
//
//   * d is a plain matrix when level == 1 and a length-`level` cell
//     array d{1..level} when level > 1. d{1} is the finest-scale detail.
//   * Empty x, scalar x, odd length, level<=0 — MATLAB throws; we mirror
//     the messages.
//
// Complex input is supported only in 'noninteger' mode (lifting floor
// is not defined on complex values; MATLAB also disallows it).

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/dwt/haart.hpp>

// Compute-only TU: Value substrate + Error, no engine. The haart builtin
// (CallContext wrapper) lives in dwt/haar_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <cmath>

namespace numkit::wavelet {

namespace {

int max_haar_level(keel::usize N)
{
    return static_cast<int>(keel::wavelet::haar_max_level(N));
}

void haar_step_real(const double *x, keel::usize inLen, double *a, double *d,
                    bool integer)
{
    const keel::usize H = inLen / 2;
    keel::wavelet::haar_step(keel::ArrayRef<keel::f64>(x, inLen),
                             keel::MutArrayRef<keel::f64>(a, H),
                             keel::MutArrayRef<keel::f64>(d, H), integer);
}

void haar_step_cplx(const keel::c64 *x, keel::usize inLen, keel::c64 *a, keel::c64 *d)
{
    const keel::usize H = inLen / 2;
    keel::wavelet::haar_step_c64(keel::ArrayRef<keel::c64>(x, inLen),
                                 keel::MutArrayRef<keel::c64>(a, H),
                                 keel::MutArrayRef<keel::c64>(d, H));
}

// Materialize x into a column-major DOUBLE buffer of (colLen × ncols).
// For a vector the whole content is one column of length N; for a
// matrix we keep the original (rows × cols) shape.
struct Buf {
    keel::Array<double> real;
    keel::Array<keel::c64> cplx;
    keel::usize colLen = 0;
    keel::usize ncols  = 0;
    bool   isComplex = false;
};

Buf load_x(const Value &x, bool /*integer_mode*/)
{
    Buf b;
    const keel::usize rows = x.dims().rows();
    const keel::usize cols = x.dims().cols();
    const keel::usize N    = rows * cols;
    const bool isVec  = (rows <= 1) || (cols <= 1);
    if (isVec) { b.colLen = N; b.ncols = 1; }
    else       { b.colLen = rows; b.ncols = cols; }

    b.isComplex = x.isComplex();
    if (b.isComplex) {
        b.cplx.resize(N);
        for (keel::usize i = 0; i < N; ++i) b.cplx[i] = x.complexElem(i);
    } else {
        b.real.resize(N);
        for (keel::usize i = 0; i < N; ++i) b.real[i] = x.elemAsDouble(i);
    }
    return b;
}

void haart_argcheck(const Value &x, int level, bool integer)
{
    if (level < 0)
        throw Error("haart: expected LEVEL to be positive",
                    0, 0, "haart", "", "numkit:haart:level");
    if (x.numel() == 0)
        throw Error("haart: expected input number 1, X, to be nonempty",
                    0, 0, "haart", "", "numkit:haart:empty");
    const keel::usize rows = x.dims().rows();
    const keel::usize cols = x.dims().cols();
    const keel::usize N    = rows * cols;
    const bool isVec  = (rows <= 1) || (cols <= 1);
    const keel::usize colLen = isVec ? N : rows;
    if (colLen < 2 || (colLen % 2) != 0)
        throw Error("haart: the data length must be even. Consider replicating "
                    "for an even length signal, or removing the last element.",
                    0, 0, "haart", "", "numkit:haart:even");
    const int max_lev = max_haar_level(colLen);
    const int lev = (level == 0) ? max_lev : level;
    if (lev > max_lev)
        throw Error(keel::format("haart: expected LEVEL to be a scalar with value <= {}", max_lev),
                    0, 0, "haart", "", "numkit:haart:level");
    if (integer && x.isComplex())
        throw Error("haart: integer mode is not supported for complex data",
                    0, 0, "haart", "", "numkit:haart:flag");
}

} // anonymous

// ── Public C++ API (see dwt/haart.hpp) ────────────────────────────────

HaartResult haart(const Value &x, int level, bool integer,
                  keel::Allocator *mr)
{
    haart_argcheck(x, level, integer);

    Buf buf = load_x(x, false);
    const keel::usize colLen = buf.colLen;
    const keel::usize ncols  = buf.ncols;

    const int max_lev = max_haar_level(colLen);
    const int lev = (level == 0) ? max_lev : level;

    const ValueType vt = buf.isComplex ? ValueType::COMPLEX : ValueType::DOUBLE;
    keel::Array<double>  curR = keel::move(buf.real);
    keel::Array<keel::c64> curC = keel::move(buf.cplx);
    keel::usize inLen = colLen;

    keel::Array<Value> details(mr);
    details.reserve(lev);
    HaartResult R;

    for (int k = 0; k < lev; ++k) {
        const keel::usize H = inLen / 2;
        Value aMat = Value::matrix(H, ncols, vt, mr);
        Value dMat = Value::matrix(H, ncols, vt, mr);

        if (buf.isComplex) {
            keel::c64 *aP = aMat.complexDataMut();
            keel::c64 *dP = dMat.complexDataMut();
            for (keel::usize c = 0; c < ncols; ++c)
                haar_step_cplx(curC.data() + c * inLen, inLen,
                               aP + c * H, dP + c * H);
        } else {
            double *aP = aMat.doubleDataMut();
            double *dP = dMat.doubleDataMut();
            for (keel::usize c = 0; c < ncols; ++c)
                haar_step_real(curR.data() + c * inLen, inLen,
                               aP + c * H, dP + c * H, integer);
        }

        if (buf.isComplex) {
            const keel::c64 *aP = aMat.complexData();
            curC.assign(keel::ArrayRef<keel::c64>(aP, H * ncols));
        } else {
            const double *aP = aMat.doubleData();
            curR.assign(keel::ArrayRef<double>(aP, H * ncols));
        }
        details.push_back(std::move(dMat));
        if (k == lev - 1) R.a = std::move(aMat);
        inLen = H;
    }

    if (lev == 1) {
        R.d = std::move(details[0]);              // plain matrix
    } else {
        Value cellOut = Value::cell(static_cast<keel::usize>(lev), 1, mr);
        for (int k = 0; k < lev; ++k)
            cellOut.cellAt(static_cast<keel::usize>(k)) = std::move(details[k]);
        R.d = std::move(cellOut);                 // d{1} is finest
    }
    return R;
}

} // namespace numkit::wavelet
