// toolboxes/wavelet/src/swt/swt.cpp
//
// Stationary (à trous) wavelet transform: same length as the input at
// every level, no decimation. The forward analyses with dilated
// filters at each level (zeros inserted between taps); the inverse is
// the *transpose* of the forward operator, divided by 2 to cancel the
// per-level redundancy.
//
// Forward (correlation, periodic boundary, dilation 2^k at level k+1):
//   a_{k+1}[n] = sum_l Lo_D[l] · a_k[(n + l·2^k) mod N]
//   d_{k+1}[n] = sum_l Hi_D[l] · a_k[(n + l·2^k) mod N]
//
// In matrix form a = H·a_prev, d = G·a_prev with H, G circulant
// shifted-stride filters. SWT preserves dim per band per level, so
// H·Hᵀ + G·Gᵀ = 2·I. Hence the exact inverse is
//   a_prev = (1/2)·(Hᵀ·a + Gᵀ·d)
// and the transpose of a "look-forward" correlation is a "look-back"
// correlation with the same coefficients:
//   a_prev[n] = (1/2)·{ sum_l Lo_D[l] · a[(n − l·2^k) mod N]
//                     + sum_l Hi_D[l] · d[(n − l·2^k) mod N] }
//
// MATLAB layout returned/consumed:
//   row 1..n : details   (level 1 = finest, level n = coarsest)
//   row n+1  : approximation at level n

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/swt/swt.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The swt / iswt /
// modwt / imodwt builtins (CallContext wrappers) live in swt/swt_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>

namespace numkit::wavelet {

namespace {

void swtAD_argcheck(const Value &x, int n,
                    keel::ArrayRef<keel::f64> Lo_D,
                    keel::ArrayRef<keel::f64> Hi_D)
{
    if (n < 1)
        throw Error("swt: level must be ≥ 1",
                    0, 0, "swt", "", "numkit:swt:level");
    if (Lo_D.size() < 2 || Lo_D.size() % 2 != 0 || Lo_D.size() != Hi_D.size())
        throw Error("swt: filter vectors must have matching even length >= 2",
                    0, 0, "swt", "", "numkit:swt:filter");
    const keel::usize N = x.numel();
    if (N > 0) {
        const keel::usize needed = keel::as<keel::usize>(1) << n;
        if (N % needed != 0)
            throw Error("swt: signal length must be divisible by 2^n",
                        0, 0, "swt", "", "numkit:swt:size");
    }
}

void modwt_argcheck(int n)
{
    if (n < 1)
        throw Error("modwt: level must be ≥ 1",
                    0, 0, "modwt", "", "numkit:modwt:level");
}

void imodwt_argcheck(const Value &swc)
{
    internal::requireWaveClass(swc, "imodwt", "W", 1,
                               internal::WvClasses::FloatOnly,
                               /*allowComplex=*/true);
    const keel::usize H = swc.dims().rows();
    const keel::usize N = swc.dims().cols();
    if (H < 2 || N < 2)
        throw Error(keel::format("imodwt: W must be at least 2-by-2 (got a {} array).",
                                 internal::shapeOf(swc)),
                    0, 0, "imodwt", "", "numkit:imodwt:size");
}

} // anonymous

std::pair<Value, Value> swtAD(const Value &x, int n,
                              keel::ArrayRef<keel::f64> Lo_D,
                              keel::ArrayRef<keel::f64> Hi_D,
                              keel::Allocator *mr)
{
    swtAD_argcheck(x, n, Lo_D, Hi_D);
    const keel::usize N = x.numel();
    if (N == 0) {
        return {Value::matrix(keel::as<keel::usize>(n), 0, ValueType::DOUBLE, mr),
                Value::matrix(keel::as<keel::usize>(n), 0, ValueType::DOUBLE, mr)};
    }

    keel::Array<keel::f64> xv(N, mr);
    for (keel::usize i = 0; i < N; ++i) xv[i] = x.elemAsDouble(i);
    auto res = keel::wavelet::swt_ad(keel::ArrayRef<keel::f64>(xv.data(), N),
                                     keel::as<keel::i32>(n), Lo_D, Hi_D, mr);
    Value swa = Value::matrix(res.levels, res.length, ValueType::DOUBLE, mr);
    Value swd = Value::matrix(res.levels, res.length, ValueType::DOUBLE, mr);
    const keel::usize total = res.levels * res.length;
    if (total > 0) {
        std::copy(res.swa_colmajor.begin(), res.swa_colmajor.end(), swa.doubleDataMut());
        std::copy(res.swd_colmajor.begin(), res.swd_colmajor.end(), swd.doubleDataMut());
    }
    return {std::move(swa), std::move(swd)};
}

std::pair<Value, Value> swtAD(const Value &x, int n, keel::StringRef wname,
                              keel::Allocator *mr)
{
    auto fb = wavelet_filters(wname);
    return swtAD(x, n, fb.Lo_D, fb.Hi_D, mr);
}

Value swt(const Value &x, int n, keel::ArrayRef<keel::f64> Lo_D,
          keel::ArrayRef<keel::f64> Hi_D, keel::Allocator *mr)
{
    if (n < 1)
        throw Error("swt: level must be ≥ 1",
                    0, 0, "swt", "", "numkit:swt:level");
    const keel::usize N = x.numel();
    if (N == 0)
        return Value::matrix(keel::as<keel::usize>(n + 1), 0,
                             ValueType::DOUBLE, mr);

    auto [swa, swd] = swtAD(x, n, Lo_D, Hi_D, mr);
    const keel::usize H = keel::as<keel::usize>(n);
    Value out = Value::matrix(keel::as<keel::usize>(n + 1), N,
                              ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize r = 0; r < H; ++r)
        for (keel::usize c = 0; c < N; ++c)
            od[c * (H + 1) + r] = swd.elemAsDouble(c * H + r);
    for (keel::usize c = 0; c < N; ++c)
        od[c * (H + 1) + H] = swa.elemAsDouble(c * H + H - 1);
    return out;
}

Value swt(const Value &x, int n, keel::StringRef wname, keel::Allocator *mr)
{
    auto fb = wavelet_filters(wname);
    return swt(x, n, fb.Lo_D, fb.Hi_D, mr);
}

namespace {

// Exact port of the reference's phase-decomposed inverse (iswt.m):
// walk levels coarsest→finest; at level k (1-based) each of the
// 2^(k-1) stride-2^(k-1) phases reconstructs via two 'per'-mode
// sub-reconstructions (even / odd positions of the phase), averaged
// with a 0.5 factor and stored back into the running approximation.
// The store is an ARRAY ASSIGNMENT: for integer classes it rounds
// half-away-from-zero and saturates per element per phase, and for
// single it narrows to float — this is the reference's integer
// wavelet arithmetic (verified on R2025b crafted uint8/int8/uint16
// cases; a plain double-then-cast kernel diverges from level 2 on).
//
// aRow: 1×N last-approximation row; its class drives the per-phase
// store rounding and the result class. dSrc: matrix whose rows
// 0..n-1 are the detail coefficients (level 1 = finest).
Value iswtPhase(Value aRow, const Value &dSrc, keel::usize n,
                keel::ArrayRef<keel::f64> LoR,
                keel::ArrayRef<keel::f64> HiR,
                keel::Allocator *mr)
{
    const keel::usize H = dSrc.dims().rows();
    const keel::usize N = aRow.numel();
    // N >= 2^n: the deepest level works on N/2^n samples per phase.
    {
        keel::usize need = 1;
        for (keel::usize k = 0; k < n; ++k) {
            need <<= 1;
            if (need > N) {
                throw Error(keel::format("iswt: the coefficients must have at least {} "
                                         "columns for {} levels (got {}).",
                                         need, n, N),
                            0, 0, "iswt", "", "numkit:iswt:expectedVector");
            }
        }
    }
    if (N == 0)
        return Value::matrix(1, 0, ValueType::DOUBLE, mr);

    const keel::usize lf = LoR.size();
    const keel::usize ext = lf / 2;
    Value a = std::move(aRow);

    // upconvLOC: dyadup(x,0,1) → 'per' extend by lf/2 → full conv →
    // keep l samples from 0-based index lf-1 (wkeep1(y,l,lf)).
    auto upconvv = [&](keel::ArrayRef<keel::f64> x,
                       keel::ArrayRef<keel::f64> f, keel::usize l) {
        const keel::usize m = x.size();
        const keel::usize un = 2 * m;             // always even → no odd-dup
        if (ext > un)
            throw Error(keel::format("iswt: the 'per' extension of a length-{} "
                                     "subsequence by {} samples exceeds its length "
                                     "(wavelet filter too long for {} columns at {} rows).",
                                     un, ext, N, H),
                        0, 0, "iswt", "", "numkit:iswt:badsubscript");
        const keel::usize en = un + 2 * ext;
        keel::Array<keel::f64> e(en, 0.0);
        for (keel::usize i = 0; i < m; ++i) e[ext + 2 * i] = x[i];
        for (keel::usize i = 0; i < ext; ++i) {
            e[i] = e[ext + ((i + un) - (ext % un)) % un];  // wrap back
            e[ext + un + i] = e[ext + (i % un)];           // wrap fwd
        }
        keel::Array<keel::f64> c(en + lf - 1, 0.0);
        for (keel::usize i = 0; i < en; ++i) {
            const keel::f64 v = e[i];
            if (v == 0.0) continue;         // dyadup zeros contribute 0
            for (keel::usize j = 0; j < lf; ++j) c[i + j] += v * f[j];
        }
        keel::Array<keel::f64> out(l, 0.0);
        for (keel::usize i = 0; i < l; ++i) out[i] = c[lf - 1 + i];
        return out;
    };

    // idwtLOC: sum of the two band reconstructions; shift=-1 rotates
    // right by one ([y(end), y(1:end-1)]).
    auto idwtLOCv = [&](keel::ArrayRef<keel::f64> xa,
                        keel::ArrayRef<keel::f64> xd,
                        keel::usize l, bool shift) {
        auto y1 = upconvv(xa, LoR, l);
        auto y2 = upconvv(xd, HiR, l);
        if (shift) {
            const keel::f64 last = y1[l - 1] + y2[l - 1];
            for (keel::usize i = l; i > 1; --i) y1[i - 1] = y1[i - 2] + y2[i - 2];
            y1[0] = last;
        } else {
            for (keel::usize i = 0; i < l; ++i) y1[i] += y2[i];
        }
        return y1;
    };

    for (keel::usize k = n; k >= 1; --k) {
        const keel::usize step = keel::as<keel::usize>(1) << (k - 1);
        for (keel::usize first = 1; first <= step; ++first) {
            const keel::usize lon = (N - first) / step + 1;
            const keel::usize mE = (lon + 1) / 2, mO = lon / 2;
            keel::Array<keel::f64> ae(mE), de(mE), ao(mO), doo(mO);
            for (keel::usize j = 0; j < lon; ++j) {
                const keel::usize col = first - 1 + j * step;
                const keel::f64 dv = dSrc.elemAsDouble(col * H + (k - 1));
                if (j % 2 == 0) { ae[j / 2] = a.elemAsDouble(col); de[j / 2] = dv; }
                else            { ao[j / 2] = a.elemAsDouble(col); doo[j / 2] = dv; }
            }
            auto x1 = idwtLOCv(ae, de, lon, false);
            auto x2 = idwtLOCv(ao, doo, lon, true);
            for (keel::usize j = 0; j < lon; ++j) {
                const keel::usize col = first - 1 + j * step;
                internal::storeElem(a, col, 0.5 * (x1[j] + x2[j]));
            }
        }
    }
    return a;
}

} // anonymous

Value iswt(const Value &swc, keel::ArrayRef<keel::f64> Lo_R,
           keel::ArrayRef<keel::f64> Hi_R, keel::Allocator *mr)
{
    if (Lo_R.size() < 2 || Lo_R.size() % 2 != 0 || Lo_R.size() != Hi_R.size())
        throw Error("iswt: filter vectors must have matching even length >= 2",
                    0, 0, "iswt", "", "numkit:iswt:filter");
    internal::requireNonempty(swc, "iswt", "SWC", 1);
    internal::requireWaveClass(swc, "iswt", "SWC", 1,
                               internal::WvClasses::IntWavelet);
    const keel::usize H = swc.dims().rows();
    const keel::usize N = swc.dims().cols();
    if (H < 2)
        return internal::copyInClass(swc, mr);      // identity (0 levels)
    const keel::usize n = H - 1;

    Value aRow = Value::matrix(1, N, swc.type(), mr);
    for (keel::usize c = 0; c < N; ++c)
        internal::storeElem(aRow, c, swc.elemAsDouble(c * H + n));
    return iswtPhase(std::move(aRow), swc, n, Lo_R, Hi_R, mr);
}

Value iswt(const Value &swc, keel::StringRef wname, keel::Allocator *mr)
{
    auto fb = wavelet_filters(wname);
    return iswt(swc, fb.Lo_R, fb.Hi_R, mr);
}

Value iswt(const Value &swa, const Value &swd,
           keel::ArrayRef<keel::f64> Lo_R,
           keel::ArrayRef<keel::f64> Hi_R,
           keel::Allocator *mr)
{
    if (Lo_R.size() < 2 || Lo_R.size() % 2 != 0 || Lo_R.size() != Hi_R.size())
        throw Error("iswt: filter vectors must have matching even length >= 2",
                    0, 0, "iswt", "", "numkit:iswt:filter");
    internal::requireNonempty(swa, "iswt", "SWA", 1);
    internal::requireWaveClass(swa, "iswt", "SWA", 1,
                               internal::WvClasses::IntWavelet);
    internal::requireNonempty(swd, "iswt", "SWD", 2);
    internal::requireWaveClass(swd, "iswt", "SWD", 2,
                               internal::WvClasses::IntWavelet);
    if (swa.dims().cols() != swd.dims().cols())
        throw Error(keel::format("iswt: the number of columns of SWA ({}) and SWD ({}) "
                                 "must be the same.",
                                 swa.dims().cols(), swd.dims().cols()),
                    0, 0, "iswt", "", "numkit:iswt:InvSWTCSize");
    const keel::usize n = swd.dims().rows();
    const keel::usize N = swd.dims().cols();

    Value aRow = Value::matrix(1, N, swa.type(), mr);
    const keel::usize aH = swa.dims().rows();
    for (keel::usize c = 0; c < N; ++c)
        internal::storeElem(aRow, c, swa.elemAsDouble(c * aH + aH - 1));
    if (n == 0)
        return aRow;
    return iswtPhase(std::move(aRow), swd, n, Lo_R, Hi_R, mr);
}

Value iswt(const Value &swa, const Value &swd, keel::StringRef wname,
           keel::Allocator *mr)
{
    auto fb = wavelet_filters(wname);
    return iswt(swa, swd, fb.Lo_R, fb.Hi_R, mr);
}

// ════════════════════════════════════════════════════════════════════
// modwt / imodwt — maximal-overlap DWT
// ════════════════════════════════════════════════════════════════════
//
// MODWT is structurally identical to SWT but uses energy-preserving
// filters h̃ = Lo_D/√2 and g̃ = Hi_D/√2. The /√2 normalisation per
// level absorbs the factor-2 redundancy that SWT pushes into the
// inverse, so the inverse here is simply two correlations summed
// (no /2 factor).
//
// Output layout (n+1) × N:
//   row 1..n : wavelet coefficients W_j (level 1 = finest)
//   row n+1  : scaling coefficients V_n
//
// Unlike SWT, MODWT does NOT require N to divide 2ⁿ — the transform
// is shift-invariant and works for any N.

Value modwt(const Value &x, int n, keel::StringRef wname, keel::Allocator *mr)
{
    modwt_argcheck(n);
    const keel::usize N = x.numel();
    if (N == 0)
        return Value::matrix(keel::as<keel::usize>(n + 1), 0,
                             ValueType::DOUBLE, mr);

    auto fb = wavelet_filters(wname);
    keel::wavelet::FilterBank kfb{std::move(fb.Lo_D), std::move(fb.Hi_D),
                                  std::move(fb.Lo_R), std::move(fb.Hi_R)};
    keel::Array<keel::f64> xv(N, mr);
    for (keel::usize i = 0; i < N; ++i) xv[i] = x.elemAsDouble(i);
    auto w = keel::wavelet::modwt(keel::ArrayRef<keel::f64>(xv.data(), N),
                                  keel::as<keel::i32>(n), kfb, mr);
    Value out = Value::matrix(keel::as<keel::usize>(n + 1), N, ValueType::DOUBLE, mr);
    if (!w.empty()) std::copy(w.begin(), w.end(), out.doubleDataMut());
    return out;
}

Value imodwt(const Value &swc, keel::StringRef wname, keel::Allocator *mr)
{
    imodwt_argcheck(swc);
    const keel::usize H = swc.dims().rows();
    const keel::usize N = swc.dims().cols();
    const keel::usize levels = H - 1;

    auto fb = wavelet_filters(wname);
    keel::wavelet::FilterBank kfb{std::move(fb.Lo_D), std::move(fb.Hi_D),
                                  std::move(fb.Lo_R), std::move(fb.Hi_R)};

    const bool cplx = swc.type() == ValueType::COMPLEX;
    keel::Array<keel::f64> wR(H * N, mr);
    for (keel::usize i = 0; i < H * N; ++i)
        wR[i] = cplx ? swc.complexElem(i).real() : swc.elemAsDouble(i);
    auto aR = keel::wavelet::imodwt(keel::ArrayRef<keel::f64>(wR.data(), H * N), levels, N, kfb, mr);

    if (!cplx) {
        Value y = Value::matrix(1, N, swc.type(), mr);
        for (keel::usize i = 0; i < N; ++i)
            internal::storeElem(y, i, aR[i]);
        return y;
    }
    keel::Array<keel::f64> wI(H * N, mr);
    for (keel::usize i = 0; i < H * N; ++i)
        wI[i] = swc.complexElem(i).imag();
    auto aI = keel::wavelet::imodwt(keel::ArrayRef<keel::f64>(wI.data(), H * N), levels, N, kfb, mr);
    Value y = Value::complexMatrix(1, N, mr);
    for (keel::usize i = 0; i < N; ++i)
        y.complexDataMut()[i] = keel::c64(aR[i], aI[i]);
    return y;
}

} // namespace numkit::wavelet
