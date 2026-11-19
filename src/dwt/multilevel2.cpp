// toolboxes/wavelet/src/dwt/multilevel2.cpp
//
// 2-D multilevel DWT family — wavedec2 / waverec2 / appcoef2 / detcoef2.
// Built on the single-level dwt2 / idwt2 primitives, exactly as the 1-D
// wavedec / waverec / appcoef / detcoef (multilevel.cpp) wrap dwt / idwt.
//
// The (C, S) packing is MATLAB-canonical (see `doc wavedec2`):
//
//   C = [cA_N | cH_N cV_N cD_N | cH_{N-1} cV_{N-1} cD_{N-1} | … |
//        cH_1 cV_1 cD_1]            (coarsest-first, each band column-major)
//   S = [ size(cA_N);
//         size(detail @ level N);   (== size(cA_N))
//         size(detail @ level N-1);
//         …
//         size(detail @ level 1);
//         size(X) ]                 ((N+2) × 2)
//
// Sizes satisfy size(A_k) = S(N-k+2, :) for k = 1..N, and size(A_N) = S(1,:).

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/wavelet/dwt/multilevel2.hpp>
#include <numkit/wavelet/dwt/dwt2.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The CallContext
// builtins live in dwt/multilevel2_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <array>
#include <utility>

namespace numkit::wavelet {

namespace {

struct Sz { keel::usize rows, cols; keel::usize numel() const { return rows * cols; } };

// Append every element of a band (column-major storage order) to `c`.
void appendBand(keel::Array<double> &c, const Value &band) {
    const keel::usize n = band.numel();
    for (keel::usize i = 0; i < n; ++i) c.push_back(band.elemAsDouble(i));
}

// Reshape a flat segment c[off, off+rows*cols) into a rows×cols matrix
// (column-major, matching nk/MATLAB storage).
Value reshapeSeg(keel::ArrayRef<double> c, keel::usize off, Sz sz,
                 keel::Allocator *mr) {
    Value m = Value::matrix(sz.rows, sz.cols, ValueType::DOUBLE, mr);
    double *d = m.doubleDataMut();
    for (keel::usize i = 0; i < sz.numel(); ++i)
        d[i] = (off + i < c.size()) ? c[off + i] : 0.0;
    return m;
}

keel::Array<double> vecFromValue(const Value &v, keel::Allocator *mr = nullptr) {
    keel::Array<double> out(v.numel(), mr);
    for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

// Read row `i` (0-indexed) of the (R×2) size matrix S as a Sz.
Sz sRow(const Value &S, keel::usize i) {
    const keel::usize R = S.dims().rows();
    return { static_cast<keel::usize>(S.elemAsDouble(i)),
             static_cast<keel::usize>(S.elemAsDouble(R + i)) };
}

void wavedec2_argcheck(int n)
{
    if (n < 1)
        throw Error("wavedec2: level must be >= 1",
                    0, 0, "wavedec2", "", "numkit:wavedec2:level");
}

int detcoef2_argcheck(keel::StringRef type, const Value &C, const Value &S, int level)
{
    internal::requireWaveClass(C, "detcoef2", "C", 2,
                               internal::WvClasses::AnyNumeric,
                               /*allowComplex=*/true);
    internal::requireWaveClass(S, "detcoef2", "S", 3,
                               internal::WvClasses::AnyNumeric);
    const int N = static_cast<int>(S.dims().rows()) - 2;
    if (N < 1)
        throw Error("detcoef2: S must have at least 3 rows (1 level)",
                    0, 0, "detcoef2", "", "numkit:detcoef2:size");
    if (level < 1 || level > N)
        throw Error("detcoef2: level out of range [1, N]",
                    0, 0, "detcoef2", "", "numkit:detcoef2:level");

    if (type == "h" || type == "H") return 0;
    if (type == "v" || type == "V") return 1;
    if (type == "d" || type == "D") return 2;
    throw Error(keel::format("detcoef2: type must be 'h', 'v' or 'd' (got '{}')", type),
                0, 0, "detcoef2", "", "numkit:detcoef2:type");
}

} // anonymous

Wavedec2Result wavedec2(const Value &X, int n, keel::StringRef wname,
                        keel::Allocator *mr)
{
    wavedec2_argcheck(n);

    const Sz xsz{ X.dims().rows(), X.dims().cols() };

    // Run n successive single-level dwt2 on the running approximation.
    // Stash each level's detail bands (finest = level 1 at index 0).
    keel::Array<Value> Hs(static_cast<keel::usize>(n), mr), Vs(static_cast<keel::usize>(n), mr), Ds(static_cast<keel::usize>(n), mr);
    keel::Array<Sz> detSize(static_cast<keel::usize>(n), mr);
    Value running = X;
    Value approx;
    for (int k = 0; k < n; ++k) {
        auto r = dwt2(running, wname, mr);
        detSize[k] = { r.cH.dims().rows(), r.cH.dims().cols() };
        Hs[k] = std::move(r.cH);
        Vs[k] = std::move(r.cV);
        Ds[k] = std::move(r.cD);
        running = std::move(r.cA);
        if (k == n - 1) approx = running;
    }
    const Sz asz{ approx.dims().rows(), approx.dims().cols() };

    // C = [cA_N | (cH cV cD)_N | … | (cH cV cD)_1]  (coarsest-first).
    keel::Array<double> C(mr);
    appendBand(C, approx);
    for (int k = n - 1; k >= 0; --k) {
        appendBand(C, Hs[k]);
        appendBand(C, Vs[k]);
        appendBand(C, Ds[k]);
    }

    // S = (N+2)×2: [size(cA_N); detail@N; detail@N-1; …; detail@1; size(X)].
    const keel::usize R = static_cast<keel::usize>(n) + 2;
    Value S = Value::matrix(R, 2, ValueType::DOUBLE, mr);
    double *sd = S.doubleDataMut();
    auto setRow = [&](keel::usize i, Sz z) { sd[i] = (double)z.rows; sd[R + i] = (double)z.cols; };
    setRow(0, asz);
    for (int k = n - 1, i = 1; k >= 0; --k, ++i) setRow((keel::usize)i, detSize[k]);
    setRow(R - 1, xsz);

    Value c = Value::matrix(1, C.size(), ValueType::DOUBLE, mr);
    if (!C.empty()) std::copy(C.begin(), C.end(), c.doubleDataMut());
    return { std::move(c), std::move(S) };
}

// Approximation at `level` (level==N → stored cA_N; level<N → reconstruct;
// level==0 → full image reconstruction). Reused by appcoef2 and waverec2.
static Value appcoef2_impl(const Value &C, const Value &S,
                           keel::StringRef wname, int level,
                           keel::Allocator *mr)
{
    const int N = static_cast<int>(S.dims().rows()) - 2;
    if (N < 1)
        throw Error("appcoef2: S must have at least 3 rows (1 level)",
                    0, 0, "appcoef2", "", "numkit:appcoef2:size");
    if (level < 0) level = N;                 // default = coarsest
    if (level < 0 || level > N)
        throw Error("appcoef2: level out of range [0, N]",
                    0, 0, "appcoef2", "", "numkit:appcoef2:level");

    auto Cv = vecFromValue(C, mr);

    // running = cA_N (first prod(S(1,:)) entries).
    const Sz asz = sRow(S, 0);
    Value running = reshapeSeg(Cv, 0, asz, mr);
    if (level == N) return running;

    // size(A_m): m==0 → S(N+2,:); else S(N-m+2,:).
    auto approxSize = [&](int m) -> Sz {
        return (m == 0) ? sRow(S, (keel::usize)N + 1) : sRow(S, (keel::usize)(N - m + 1));
    };

    // Walk down from level N to `level`, consuming detail blocks coarsest-first.
    keel::usize off = asz.numel();
    for (int lev = N; lev > level; --lev) {
        const Sz dsz = sRow(S, (keel::usize)(N - lev + 1));   // detail size at `lev`
        const keel::usize dn = dsz.numel();
        Value cH = reshapeSeg(Cv, off, dsz, mr);          off += dn;
        Value cV = reshapeSeg(Cv, off, dsz, mr);          off += dn;
        Value cD = reshapeSeg(Cv, off, dsz, mr);          off += dn;
        const Sz tgt = approxSize(lev - 1);               // size of A_{lev-1}
        running = idwt2(running, cH, cV, cD, wname,
                        (long long)tgt.rows, (long long)tgt.cols, mr);
    }
    return running;
}

Value waverec2(const Value &C, const Value &S, keel::StringRef wname,
               keel::Allocator *mr)
{
    return appcoef2_impl(C, S, wname, /*level=*/0, mr);
}

Value appcoef2(const Value &C, const Value &S, keel::StringRef wname,
               int level, keel::Allocator *mr)
{
    return appcoef2_impl(C, S, wname, level, mr);
}

Value detcoef2(keel::StringRef type, const Value &C, const Value &S,
               int level, keel::Allocator *mr)
{
    const int which = detcoef2_argcheck(type, C, S, level);
    const int N = static_cast<int>(S.dims().rows()) - 2;

    // Offset: skip cA_N, then the three detail blocks for levels N..level+1.
    keel::usize off = sRow(S, 0).numel();
    for (int lev = N; lev > level; --lev)
        off += 3 * sRow(S, (keel::usize)(N - lev + 1)).numel();

    const Sz dsz = sRow(S, (keel::usize)(N - level + 1));   // detail size at `level`
    off += static_cast<keel::usize>(which) * dsz.numel();

    // Class-preserving gather straight out of C.
    const bool cplx = C.type() == ValueType::COMPLEX;
    Value out = cplx ? Value::complexMatrix(dsz.rows, dsz.cols, mr)
                     : Value::matrix(dsz.rows, dsz.cols, C.type(), mr);
    for (keel::usize i = 0; i < dsz.numel(); ++i) {
        const keel::usize src = off + i;
        if (src >= C.numel())
            throw Error(keel::format("detcoef2: C is too short for the sizes in S "
                                     "(needed {} elements, got {}).",
                                     off + dsz.numel(), C.numel()),
                        0, 0, "detcoef2", "", "numkit:detcoef2:size");
        if (cplx) out.complexDataMut()[i] = C.complexElem(src);
        else internal::storeElem(out, i, C.elemAsDouble(src));
    }
    return out;
}

} // namespace numkit::wavelet
