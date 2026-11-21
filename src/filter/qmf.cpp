// toolboxes/wavelet/src/filter/qmf.cpp
//
// Wavelet filter helpers: qmf, wrev.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/ref.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/filter/qmf.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The wrev / qmf
// builtins (CallContext wrappers) live in filter/qmf_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::wavelet {

namespace {

// Read shape (rows, cols). For non-2D arrays we treat them as flat row.
void readShape(const Value &x, keel::usize &rows, keel::usize &cols)
{
    rows = x.dims().rows();
    cols = x.dims().cols();
    if (rows == 0 && cols == 0) { rows = 1; cols = 0; }
}

Value sameShape(keel::Allocator *mr,
                keel::usize rows, keel::usize cols)
{
    return Value::matrix(rows, cols, ValueType::DOUBLE, mr);
}

void wrev_argcheck(const Value &x)
{
    internal::requireWaveClass(x, "wrev", "X", 1,
                               internal::WvClasses::AnyWithChar,
                               /*allowComplex=*/true);
}

} // anonymous

// ── Public C++ API (see filter/qmf.hpp) ───────────────────────────────

// wrev: reverse along the first non-singleton dimension. Row/col vector ->
// element reverse; matrix (M×N) -> per-column reverse (= flipud). Probed
// R2025b: the reference takes EVERY array class — single, logical, uint8,
// int16, char, complex — and the result keeps the input's class (the old
// kernel returned double for every real input, silently losing the class).
Value wrev(const Value &x, keel::Allocator *mr)
{
    wrev_argcheck(x);
    keel::usize rows, cols;
    readShape(x, rows, cols);
    const keel::usize N = rows * cols;
    const bool cplx = x.type() == ValueType::COMPLEX;
    Value y = cplx ? Value::complexMatrix(rows, cols, mr)
                   : internal::allocLike(x, x.type(), mr);
    if (N == 0) return y;

    const bool isRowVec = (rows == 1);
    if (isRowVec) {
        if (cplx) {
            const keel::c64 *src = x.complexData();
            keel::c64 *dst = y.complexDataMut();
            for (keel::usize i = 0; i < N; ++i) dst[i] = src[N - 1 - i];
        } else {
            for (keel::usize i = 0; i < N; ++i)
                internal::storeElem(y, i, x.elemAsDouble(N - 1 - i));
        }
    } else {
        // Column-major: column c starts at offset c*rows; reverse each.
        if (cplx) {
            const keel::c64 *src = x.complexData();
            keel::c64 *dst = y.complexDataMut();
            for (keel::usize c = 0; c < cols; ++c) {
                const keel::usize base = c * rows;
                for (keel::usize r = 0; r < rows; ++r)
                    dst[base + r] = src[base + (rows - 1 - r)];
            }
        } else {
            for (keel::usize c = 0; c < cols; ++c) {
                const keel::usize base = c * rows;
                for (keel::usize r = 0; r < rows; ++r)
                    internal::storeElem(y, base + r,
                                        x.elemAsDouble(base + (rows - 1 - r)));
            }
        }
    }
    return y;
}

// qmf: quadrature mirror filter. y(k) = (-1)^(k-1) * x(N-k+1) for p == 0
// (default); the whole result is negated for p == 1.
//   qmf([1 2 3 4]) -> [4 -3 2 -1]; qmf([1 2 3 4], 1) -> [-4 3 -2 1].
Value qmf(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    keel::usize rows, cols;
    readShape(x, rows, cols);
    const keel::usize N = rows * cols;
    Value y = sameShape(mr, rows, cols);
    if (N == 0) return y;
    keel::Array<keel::f64> xv(N, mr);
    for (keel::usize k = 0; k < N; ++k) xv[k] = x.elemAsDouble(k);
    keel::wavelet::qmf(keel::ArrayRef<keel::f64>(xv.data(), N),
                       keel::MutArrayRef<keel::f64>(y.doubleDataMut(), N),
                       keel::as<keel::i32>(p));
    return y;
}

} // namespace numkit::wavelet
