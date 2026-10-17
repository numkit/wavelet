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

#include <numkit/wavelet/swt/swt.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The swt / iswt /
// modwt / imodwt builtins (CallContext wrappers) live in swt/swt_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace numkit::wavelet {

namespace {

// y[i] = sum_j h[j] · x[(i + j·step) mod N]  (forward / analysis)
std::vector<double> circ_corr_fwd(const std::vector<double> &x,
                                  const std::vector<double> &h,
                                  size_t step)
{
    const size_t N = x.size();
    std::vector<double> y(N, 0.0);
    if (N == 0 || h.empty()) return y;
    for (size_t i = 0; i < N; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < h.size(); ++j) {
            const size_t idx = (i + j * step) % N;
            acc += h[j] * x[idx];
        }
        y[i] = acc;
    }
    return y;
}

// y[i] = sum_j h[j] · x[(i − j·step) mod N]  (transpose of fwd)
std::vector<double> circ_corr_back(const std::vector<double> &x,
                                   const std::vector<double> &h,
                                   size_t step)
{
    const size_t N = x.size();
    std::vector<double> y(N, 0.0);
    if (N == 0 || h.empty()) return y;
    for (size_t i = 0; i < N; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < h.size(); ++j) {
            // Modular subtract: ((i − j·step) mod N) ≡ ((i + N − (j·step mod N)) mod N).
            const size_t shift = (j * step) % N;
            const size_t idx = (i + N - shift) % N;
            acc += h[j] * x[idx];
        }
        y[i] = acc;
    }
    return y;
}

} // anonymous

std::pair<Value, Value> swtAD(const Value &x, int n,
                              const std::vector<double> &Lo_D,
                              const std::vector<double> &Hi_D,
                              std::pmr::memory_resource *mr)
{
    if (n < 1)
        throw Error("swt: level must be ≥ 1",
                    0, 0, "swt", "", "numkit:swt:level");
    if (Lo_D.size() < 2 || Lo_D.size() % 2 != 0 || Lo_D.size() != Hi_D.size())
        throw Error("swt: filter vectors must have matching even length >= 2",
                    0, 0, "swt", "", "numkit:swt:filter");
    const size_t N = x.numel();
    if (N == 0) {
        return {Value::matrix(static_cast<size_t>(n), 0, ValueType::DOUBLE, mr),
                Value::matrix(static_cast<size_t>(n), 0, ValueType::DOUBLE, mr)};
    }

    const size_t needed = static_cast<size_t>(1) << n;
    if (N % needed != 0)
        throw Error("swt: signal length must be divisible by 2^n",
                    0, 0, "swt", "", "numkit:swt:size");

    std::vector<double> xv(N);
    for (size_t i = 0; i < N; ++i) xv[i] = x.elemAsDouble(i);
    std::vector<double> tempLo = Lo_D, tempHi = Hi_D;

    Value swa = Value::matrix(static_cast<size_t>(n), N,
                              ValueType::DOUBLE, mr);
    Value swd = Value::matrix(static_cast<size_t>(n), N,
                              ValueType::DOUBLE, mr);
    double *ad = swa.doubleDataMut();
    double *dd = swd.doubleDataMut();
    const size_t H = static_cast<size_t>(n);

    auto convKeep = [&](const std::vector<double> &xe,
                        const std::vector<double> &h, size_t lf) {
        std::vector<double> out(N, 0.0);
        for (size_t j = 0; j < lf; ++j) {
            const double hj = h[j];
            if (hj == 0.0) continue;     // dyadup'd zero taps
            for (size_t m = 0; m < N; ++m)
                out[m] += hj * xe[lf + m - j];
        }
        return out;
    };

    for (int k = 0; k < n; ++k) {
        const size_t lf = tempLo.size();
        const size_t ext = lf / 2;
        std::vector<double> xe(N + 2 * ext);
        for (size_t i = 0; i < ext; ++i) {
            xe[i] = xv[(N + i - ext) % N];
            xe[ext + N + i] = xv[i % N];
        }
        std::copy(xv.begin(), xv.end(), xe.begin() + ext);
        auto dRow = convKeep(xe, tempHi, lf);
        auto aRow = convKeep(xe, tempLo, lf);
        for (size_t c = 0; c < N; ++c) {
            ad[c * H + k] = aRow[c];    // approximation at level k+1
            dd[c * H + k] = dRow[c];    // detail at level k+1
        }
        xv = std::move(aRow);
        auto up = [](const std::vector<double> &h) {
            std::vector<double> u(2 * h.size(), 0.0);
            for (size_t t = 0; t < h.size(); ++t) u[2 * t] = h[t];
            return u;
        };
        tempLo = up(tempLo);
        tempHi = up(tempHi);
    }
    return {std::move(swa), std::move(swd)};
}

std::pair<Value, Value> swtAD(const Value &x, int n, const std::string &wname,
                              std::pmr::memory_resource *mr)
{
    auto fb = wavelet_filters(wname);
    return swtAD(x, n, fb.Lo_D, fb.Hi_D, mr);
}

Value swt(const Value &x, int n, const std::vector<double> &Lo_D,
          const std::vector<double> &Hi_D, std::pmr::memory_resource *mr)
{
    if (n < 1)
        throw Error("swt: level must be ≥ 1",
                    0, 0, "swt", "", "numkit:swt:level");
    const size_t N = x.numel();
    if (N == 0)
        return Value::matrix(static_cast<size_t>(n + 1), 0,
                             ValueType::DOUBLE, mr);

    auto [swa, swd] = swtAD(x, n, Lo_D, Hi_D, mr);
    const size_t H = static_cast<size_t>(n);
    Value out = Value::matrix(static_cast<size_t>(n + 1), N,
                              ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (size_t r = 0; r < H; ++r)
        for (size_t c = 0; c < N; ++c)
            od[c * (H + 1) + r] = swd.elemAsDouble(c * H + r);
    for (size_t c = 0; c < N; ++c)
        od[c * (H + 1) + H] = swa.elemAsDouble(c * H + H - 1);
    return out;
}

Value swt(const Value &x, int n, const std::string &wname, std::pmr::memory_resource *mr)
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
Value iswtPhase(Value aRow, const Value &dSrc, size_t n,
                const std::vector<double> &LoR,
                const std::vector<double> &HiR,
                std::pmr::memory_resource *mr)
{
    const size_t H = dSrc.dims().rows();
    const size_t N = aRow.numel();
    // N >= 2^n: the deepest level works on N/2^n samples per phase.
    {
        size_t need = 1;
        for (size_t k = 0; k < n; ++k) {
            need <<= 1;
            if (need > N) {
                throw Error("iswt: the coefficients must have at least "
                                + std::to_string(need)
                                + " columns for " + std::to_string(n)
                                + " levels (got " + std::to_string(N) + ").",
                            0, 0, "iswt", "", "numkit:iswt:expectedVector");
            }
        }
    }
    if (N == 0)
        return Value::matrix(1, 0, ValueType::DOUBLE, mr);

    const size_t lf = LoR.size();
    const size_t ext = lf / 2;
    Value a = std::move(aRow);

    // upconvLOC: dyadup(x,0,1) → 'per' extend by lf/2 → full conv →
    // keep l samples from 0-based index lf-1 (wkeep1(y,l,lf)).
    auto upconvv = [&](const std::vector<double> &x,
                       const std::vector<double> &f, size_t l) {
        const size_t m = x.size();
        const size_t un = 2 * m;             // always even → no odd-dup
        if (ext > un)
            throw Error("iswt: the 'per' extension of a length-"
                            + std::to_string(un)
                            + " subsequence by " + std::to_string(ext)
                            + " samples exceeds its length (wavelet filter "
                            + "too long for " + std::to_string(N)
                            + " columns at " + std::to_string(H) + " rows).",
                        0, 0, "iswt", "", "numkit:iswt:badsubscript");
        const size_t en = un + 2 * ext;
        std::vector<double> e(en, 0.0);
        for (size_t i = 0; i < m; ++i) e[ext + 2 * i] = x[i];
        for (size_t i = 0; i < ext; ++i) {
            e[i] = e[ext + ((i + un) - (ext % un)) % un];  // wrap back
            e[ext + un + i] = e[ext + (i % un)];           // wrap fwd
        }
        std::vector<double> c(en + lf - 1, 0.0);
        for (size_t i = 0; i < en; ++i) {
            const double v = e[i];
            if (v == 0.0) continue;         // dyadup zeros contribute 0
            for (size_t j = 0; j < lf; ++j) c[i + j] += v * f[j];
        }
        std::vector<double> out(l, 0.0);
        for (size_t i = 0; i < l; ++i) out[i] = c[lf - 1 + i];
        return out;
    };

    // idwtLOC: sum of the two band reconstructions; shift=-1 rotates
    // right by one ([y(end), y(1:end-1)]).
    auto idwtLOCv = [&](const std::vector<double> &xa,
                        const std::vector<double> &xd,
                        size_t l, bool shift) {
        auto y1 = upconvv(xa, LoR, l);
        auto y2 = upconvv(xd, HiR, l);
        if (shift) {
            const double last = y1[l - 1] + y2[l - 1];
            for (size_t i = l; i > 1; --i) y1[i - 1] = y1[i - 2] + y2[i - 2];
            y1[0] = last;
        } else {
            for (size_t i = 0; i < l; ++i) y1[i] += y2[i];
        }
        return y1;
    };

    for (size_t k = n; k >= 1; --k) {
        const size_t step = static_cast<size_t>(1) << (k - 1);
        for (size_t first = 1; first <= step; ++first) {
            const size_t lon = (N - first) / step + 1;
            const size_t mE = (lon + 1) / 2, mO = lon / 2;
            std::vector<double> ae(mE), de(mE), ao(mO), doo(mO);
            for (size_t j = 0; j < lon; ++j) {
                const size_t col = first - 1 + j * step;
                const double dv = dSrc.elemAsDouble(col * H + (k - 1));
                if (j % 2 == 0) { ae[j / 2] = a.elemAsDouble(col); de[j / 2] = dv; }
                else            { ao[j / 2] = a.elemAsDouble(col); doo[j / 2] = dv; }
            }
            auto x1 = idwtLOCv(ae, de, lon, false);
            auto x2 = idwtLOCv(ao, doo, lon, true);
            for (size_t j = 0; j < lon; ++j) {
                const size_t col = first - 1 + j * step;
                internal::storeElem(a, col, 0.5 * (x1[j] + x2[j]));
            }
        }
    }
    return a;
}

} // anonymous

Value iswt(const Value &swc, const std::vector<double> &Lo_R,
           const std::vector<double> &Hi_R, std::pmr::memory_resource *mr)
{
    if (Lo_R.size() < 2 || Lo_R.size() % 2 != 0 || Lo_R.size() != Hi_R.size())
        throw Error("iswt: filter vectors must have matching even length >= 2",
                    0, 0, "iswt", "", "numkit:iswt:filter");
    internal::requireNonempty(swc, "iswt", "SWC", 1);
    internal::requireWaveClass(swc, "iswt", "SWC", 1,
                               internal::WvClasses::IntWavelet);
    const size_t H = swc.dims().rows();
    const size_t N = swc.dims().cols();
    if (H < 2)
        return internal::copyInClass(swc, mr);      // identity (0 levels)
    const size_t n = H - 1;

    Value aRow = Value::matrix(1, N, swc.type(), mr);
    for (size_t c = 0; c < N; ++c)
        internal::storeElem(aRow, c, swc.elemAsDouble(c * H + n));
    return iswtPhase(std::move(aRow), swc, n, Lo_R, Hi_R, mr);
}

Value iswt(const Value &swc, const std::string &wname, std::pmr::memory_resource *mr)
{
    auto fb = wavelet_filters(wname);
    return iswt(swc, fb.Lo_R, fb.Hi_R, mr);
}

Value iswt(const Value &swa, const Value &swd,
           const std::vector<double> &Lo_R,
           const std::vector<double> &Hi_R,
           std::pmr::memory_resource *mr)
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
        throw Error("iswt: the number of columns of SWA ("
                        + std::to_string(swa.dims().cols())
                        + ") and SWD (" + std::to_string(swd.dims().cols())
                        + ") must be the same.",
                    0, 0, "iswt", "", "numkit:iswt:InvSWTCSize");
    const size_t n = swd.dims().rows();
    const size_t N = swd.dims().cols();

    Value aRow = Value::matrix(1, N, swa.type(), mr);
    const size_t aH = swa.dims().rows();
    for (size_t c = 0; c < N; ++c)
        internal::storeElem(aRow, c, swa.elemAsDouble(c * aH + aH - 1));
    if (n == 0)
        return aRow;
    return iswtPhase(std::move(aRow), swd, n, Lo_R, Hi_R, mr);
}

Value iswt(const Value &swa, const Value &swd, const std::string &wname,
           std::pmr::memory_resource *mr)
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

Value modwt(const Value &x, int n, const std::string &wname, std::pmr::memory_resource *mr)
{
    if (n < 1)
        throw Error("modwt: level must be ≥ 1",
                    0, 0, "modwt", "", "numkit:modwt:level");
    const size_t N = x.numel();
    if (N == 0)
        return Value::matrix(static_cast<size_t>(n + 1), 0,
                             ValueType::DOUBLE, mr);

    auto fb = wavelet_filters(wname);
    // Pre-scale filters by 1/√2 once (so the inner correlation is a
    // straight sum without per-tap multiplications).
    // MODWT (Percival & Walden): W_{j,t} = Σ_l h̃_l X_{(t-l) mod N} with
    // the MODWT filters h̃ = wrev(Hi_D)/√2, g̃ = wrev(Lo_D)/√2 (the
    // analysis filters are TIME-REVERSED relative to wfilters' Hi_D/Lo_D,
    // then applied as a look-back circular convolution). This matches
    // MATLAB R2025b modwt (e.g. Haar W_{1,1} wraps to 0.5·(x1−xN)).
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    std::vector<double> hLo = fb.Lo_D, hHi = fb.Hi_D;
    std::reverse(hLo.begin(), hLo.end());
    std::reverse(hHi.begin(), hHi.end());
    for (auto &v : hLo) v *= inv_sqrt2;
    for (auto &v : hHi) v *= inv_sqrt2;

    std::vector<double> a(N);
    for (size_t i = 0; i < N; ++i) a[i] = x.elemAsDouble(i);

    Value out = Value::matrix(static_cast<size_t>(n + 1), N,
                              ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    const size_t H = static_cast<size_t>(n + 1);

    for (int k = 0; k < n; ++k) {
        const size_t step = static_cast<size_t>(1) << k;
        auto a_next = circ_corr_back(a, hLo, step);
        auto d_next = circ_corr_back(a, hHi, step);
        for (size_t c = 0; c < N; ++c) od[c * H + k] = d_next[c];
        a = std::move(a_next);
    }
    for (size_t c = 0; c < N; ++c) od[c * H + n] = a[c];
    return out;
}

Value imodwt(const Value &swc, const std::string &wname, std::pmr::memory_resource *mr)
{
    // Probed R2025b: w is double/single only (the integer classes and
    // logical reject with the toolbox's InvalidCFSSize diagnostic) but
    // complex w computes a complex result; both H and N must be >= 2.
    internal::requireWaveClass(swc, "imodwt", "W", 1,
                               internal::WvClasses::FloatOnly,
                               /*allowComplex=*/true);
    const size_t H = swc.dims().rows();
    const size_t N = swc.dims().cols();
    if (H < 2 || N < 2)
        throw Error("imodwt: W must be at least 2-by-2 (got a "
                        + internal::shapeOf(swc) + " array).",
                    0, 0, "imodwt", "", "numkit:imodwt:size");
    const int n = static_cast<int>(H) - 1;

    auto fb = wavelet_filters(wname);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    // Use the same reversed MODWT filters as the forward; the inverse is
    // its exact transpose, so the look-back analysis becomes a look-forward
    // synthesis (no /2 — the /√2 in forward absorbed the redundancy).
    std::vector<double> hLo = fb.Lo_D, hHi = fb.Hi_D;
    std::reverse(hLo.begin(), hLo.end());
    std::reverse(hHi.begin(), hHi.end());
    for (auto &v : hLo) v *= inv_sqrt2;
    for (auto &v : hHi) v *= inv_sqrt2;

    const bool cplx = swc.type() == ValueType::COMPLEX;
    // The inverse is linear in the coefficients: run the real and
    // imaginary components through the same double kernel.
    auto readCoef = [&](std::size_t row, std::size_t col, bool imag) -> double {
        const std::size_t flat = col * H + row;
        if (!cplx)
            return swc.elemAsDouble(flat);
        return imag ? swc.complexElem(flat).imag()
                    : swc.complexElem(flat).real();
    };
    auto modwtPass = [&](std::vector<double> a, bool imag) {
        for (int k = n - 1; k >= 0; --k) {
            const size_t step = static_cast<size_t>(1) << k;
            std::vector<double> d(N);
            for (size_t i = 0; i < N; ++i)
                d[i] = readCoef(static_cast<size_t>(k), i, imag);
            auto lo = circ_corr_fwd(a, hLo, step);
            auto hi = circ_corr_fwd(d, hHi, step);
            for (size_t i = 0; i < N; ++i) a[i] = lo[i] + hi[i];
        }
        return a;
    };

    std::vector<double> aR(N);
    for (size_t i = 0; i < N; ++i)
        aR[i] = readCoef(static_cast<size_t>(n), i, /*imag=*/false);
    aR = modwtPass(std::move(aR), /*imag=*/false);

    if (!cplx) {
        Value y = Value::matrix(1, N, swc.type(), mr);
        for (size_t i = 0; i < N; ++i)
            internal::storeElem(y, i, aR[i]);
        return y;
    }
    std::vector<double> aI(N);
    for (size_t i = 0; i < N; ++i)
        aI[i] = readCoef(static_cast<size_t>(n), i, /*imag=*/true);
    aI = modwtPass(std::move(aI), /*imag=*/true);
    Value y = Value::complexMatrix(1, N, mr);
    for (size_t i = 0; i < N; ++i)
        y.complexDataMut()[i] = Complex(aR[i], aI[i]);
    return y;
}

} // namespace numkit::wavelet
