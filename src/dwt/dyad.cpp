// toolboxes/wavelet/src/dwt/dyad.cpp
//
// Dyadic upsample/downsample helpers and wmaxlev.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/utf8.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/dwt/dyad.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>

// Compute-only TU: Value substrate + Error, no engine. The dyaddown /
// dyadup / wmaxlev builtins (CallContext wrappers) live in dwt/dyad_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <cmath>

namespace numkit::wavelet {

namespace {

char dyad_argcheck(const Value & /*x*/, keel::i32 /*odd*/, char type)
{
    const char c = keel::to_ascii_lower(type);
    if (c != 'c' && c != 'r' && c != 'm')
        throw Error("dyad: type must be 'c', 'r', or 'm'",
                    0, 0, "dyad", "", "numkit:dyad:type");
    return c;
}

void wmaxlev_argcheck(const Value &N, keel::StringRef /*wname*/)
{
    if (N.numel() == 0)
        throw Error("wmaxlev: N must not be empty",
                    0, 0, "wmaxlev", "", "numkit:wmaxlev:empty");
}

void readShape(const Value &x, keel::usize &rows, keel::usize &cols)
{
    rows = x.dims().rows();
    cols = x.dims().cols();
    if (rows == 0 && cols == 0) { rows = 1; cols = 0; }
}

// Return shape with dimension `outN` along whichever axis was the
// non-singleton in the input. For column inputs (rows>1, cols==1)
// the output is M×1; otherwise 1×M.
void outShape(keel::usize inRows, keel::usize inCols, keel::usize outN,
              keel::usize &outRows, keel::usize &outCols)
{
    if (inCols == 1 && inRows > 1) { outRows = outN; outCols = 1; }
    else                            { outRows = 1;   outCols = outN; }
}

inline keel::usize outLenDown(keel::usize N, keel::i32 odd)
{
    return (odd == 0) ? (N / 2) : ((N + 1) / 2);
}
inline keel::usize outLenUp(keel::usize N, keel::i32 odd)
{
    if (N == 0) return 0;
    return (odd == 0) ? (2 * N - 1) : (2 * N + 1);
}
inline bool isMatrix(keel::usize rows, keel::usize cols)
{
    return rows > 1 && cols > 1;
}

} // anonymous

// ── Public C++ API (see dwt/dyad.hpp) ─────────────────────────────────

Value dyaddown(const Value &x, keel::i32 odd, char type,
               keel::Allocator *mr)
{
    type = dyad_argcheck(x, odd, type);
    keel::usize rows, cols;
    readShape(x, rows, cols);

    if (!isMatrix(rows, cols)) {
        const keel::usize N = rows * cols;
        const keel::usize outN = outLenDown(N, odd);
        keel::usize outRows, outCols;
        outShape(rows, cols, outN, outRows, outCols);
        Value y = Value::matrix(outRows, outCols, ValueType::DOUBLE, mr);
        if (outN == 0) return y;
        keel::f64 *yd = y.doubleDataMut();
        const keel::usize start = (odd == 0) ? 1 : 0;
        for (keel::usize k = 0; k < outN; ++k)
            yd[k] = x.elemAsDouble(start + 2 * k);
        return y;
    }

    // 2-D matrix: column-major data layout.
    const keel::usize startC = (odd == 0) ? 1 : 0;
    const keel::usize startR = (odd == 0) ? 1 : 0;
    const keel::usize outC   = outLenDown(cols, odd);
    const keel::usize outR   = outLenDown(rows, odd);

    keel::usize resR, resC;
    if (type == 'c') { resR = rows; resC = outC; }
    else if (type == 'r') { resR = outR; resC = cols; }
    else { resR = outR; resC = outC; }

    Value y = Value::matrix(resR, resC, ValueType::DOUBLE, mr);
    if (resR == 0 || resC == 0) return y;
    keel::f64 *yd = y.doubleDataMut();

    if (type == 'c') {
        for (keel::usize kc = 0; kc < outC; ++kc) {
            const keel::usize srcC = startC + 2 * kc;
            for (keel::usize r = 0; r < rows; ++r)
                yd[kc * resR + r] = x.elemAsDouble(srcC * rows + r);
        }
    } else if (type == 'r') {
        for (keel::usize c = 0; c < cols; ++c) {
            for (keel::usize kr = 0; kr < outR; ++kr) {
                const keel::usize srcR = startR + 2 * kr;
                yd[c * resR + kr] = x.elemAsDouble(c * rows + srcR);
            }
        }
    } else {  // 'm'
        for (keel::usize kc = 0; kc < outC; ++kc) {
            const keel::usize srcC = startC + 2 * kc;
            for (keel::usize kr = 0; kr < outR; ++kr) {
                const keel::usize srcR = startR + 2 * kr;
                yd[kc * resR + kr] = x.elemAsDouble(srcC * rows + srcR);
            }
        }
    }
    return y;
}

Value dyadup(const Value &x, keel::i32 odd, char type,
             keel::Allocator *mr)
{
    type = dyad_argcheck(x, odd, type);
    keel::usize rows, cols;
    readShape(x, rows, cols);

    if (!isMatrix(rows, cols)) {
        const keel::usize N = rows * cols;
        const keel::usize outN = outLenUp(N, odd);
        keel::usize outRows, outCols;
        outShape(rows, cols, outN, outRows, outCols);
        Value y = Value::matrix(outRows, outCols, ValueType::DOUBLE, mr);
        if (outN == 0) return y;
        keel::f64 *yd = y.doubleDataMut();
        const keel::usize start = (odd == 0) ? 0 : 1;
        for (keel::usize k = 0; k < outN; ++k) yd[k] = 0.0;
        for (keel::usize i = 0; i < N; ++i)
            yd[start + 2 * i] = x.elemAsDouble(i);
        return y;
    }

    // 2-D matrix: insert zeros along the chosen axis.
    const keel::usize outC   = outLenUp(cols, odd);
    const keel::usize outR   = outLenUp(rows, odd);
    const keel::usize startC = (odd == 0) ? 0 : 1;
    const keel::usize startR = (odd == 0) ? 0 : 1;

    keel::usize resR, resC;
    if (type == 'c') { resR = rows; resC = outC; }
    else if (type == 'r') { resR = outR; resC = cols; }
    else { resR = outR; resC = outC; }
    Value y = Value::matrix(resR, resC, ValueType::DOUBLE, mr);
    if (resR == 0 || resC == 0) return y;
    keel::f64 *yd = y.doubleDataMut();
    for (keel::usize i = 0; i < resR * resC; ++i) yd[i] = 0.0;

    if (type == 'c') {
        for (keel::usize i = 0; i < cols; ++i) {
            const keel::usize dstC = startC + 2 * i;
            for (keel::usize r = 0; r < rows; ++r)
                yd[dstC * resR + r] = x.elemAsDouble(i * rows + r);
        }
    } else if (type == 'r') {
        for (keel::usize c = 0; c < cols; ++c) {
            for (keel::usize i = 0; i < rows; ++i) {
                const keel::usize dstR = startR + 2 * i;
                yd[c * resR + dstR] = x.elemAsDouble(c * rows + i);
            }
        }
    } else {  // 'm'
        for (keel::usize i = 0; i < cols; ++i) {
            const keel::usize dstC = startC + 2 * i;
            for (keel::usize k = 0; k < rows; ++k) {
                const keel::usize dstR = startR + 2 * k;
                yd[dstC * resR + dstR] = x.elemAsDouble(i * rows + k);
            }
        }
    }
    return y;
}

Value wmaxlev(const Value &N, keel::StringRef wname,
              keel::Allocator *mr)
{
    wmaxlev_argcheck(N, wname);
    // Accept scalar N or a 1×k vector; use min(N) as the effective length.
    const keel::usize k = N.numel();
    keel::f64 Nmin = N.elemAsDouble(0);
    for (keel::usize i = 1; i < k; ++i) {
        const keel::f64 v = N.elemAsDouble(i);
        if (v < Nmin) Nmin = v;
    }

    auto fb = wavelet_filters(wname);
    const keel::usize Lf = fb.Lo_D.size();
    if (Lf < 2)
        throw Error("wmaxlev: filter length must be ≥ 2",
                    0, 0, "wmaxlev", "", "numkit:wmaxlev:filter");

    const keel::usize nmin_u = (Nmin > 0.0) ? keel::as<keel::usize>(Nmin) : 0;
    const keel::f64 L = keel::as<keel::f64>(keel::wavelet::wmaxlev(nmin_u, Lf));
    return Value::scalar(L, mr);
}

} // namespace numkit::wavelet
