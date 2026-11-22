// toolboxes/wavelet/src/denoise/denoise.cpp
//
// Wavelet thresholding primitives (wthresh / wnoisest) and the
// composite VisuShrink-style denoiser (wdenoise). All built on the
// existing wavedec / waverec / detcoef plumbing — no extra DSP
// machinery here, just thresholding and a robust σ estimate.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/denoise/denoise.hpp>
#include <numkit/wavelet/dwt/multilevel.hpp>
#include <numkit/wavelet/dwt/dwt.hpp>     // ddencmp: 1-level dwt noise estimate
#include <numkit/wavelet/filter/wfilters.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The wthresh /
// wnoisest / wdenoise builtins (CallContext wrappers) live in
// denoise/denoise_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>

namespace numkit::wavelet {

namespace {

keel::f64 median_abs(keel::Array<keel::f64> v) {
    if (v.empty()) return 0.0;
    for (auto &x : v) x = keel::abs(x);
    std::sort(v.begin(), v.end());
    const keel::usize n = v.size();
    return (n % 2 == 1)
               ? v[n / 2]
               : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

keel::Array<keel::f64> vecFromValue(const Value &v, keel::Allocator *mr = nullptr) {
    keel::Array<keel::f64> out(v.numel(), mr);
    for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

void wthresh_argcheck(const Value &X, keel::StringRef sorh, const Value &T)
{
    internal::requireWaveClass(X, "wthresh", "X", 1,
                               internal::WvClasses::FloatOnly,
                               /*allowComplex=*/true);
    const bool hard = (sorh.equals_ignore_ascii_case("h"));
    const bool soft = (sorh.equals_ignore_ascii_case("s"));
    if (!hard && !soft)
        throw Error(keel::format("wthresh: SORH must be 's' (soft) or 'h' (hard) — got '{}'", sorh),
                    0, 0, "wthresh", "", "numkit:wthresh:sorh");
    if (T.type() == ValueType::COMPLEX)
        throw Error("wthresh: input 3 (T) must be real.",
                    0, 0, "wthresh", "", "numkit:wthresh:mustBeReal");
    const bool elemT = T.numel() != 1;
    if (elemT && (T.dims().rows() != X.dims().rows()
                  || T.dims().cols() != X.dims().cols()))
        throw Error(keel::format("wthresh: input 3 (T) must be a scalar or the same "
                                 "size as X (got a {} {} array for a {} X).",
                                 internal::shapeOf(T), mtypeName(T.type()), internal::shapeOf(X)),
                    0, 0, "wthresh", "", "numkit:wthresh:size");
}

keel::String wentropy_argcheck(keel::StringRef type, keel::f64 param)
{
    keel::String t(type);
    t.to_ascii_lower_in_place();
    if (t == "norm") {
        if (param < 1.0)
            throw Error("wentropy: 'norm' exponent P must be >= 1",
                        0, 0, "wentropy", "", "numkit:wentropy:norm");
    } else if (t != "shannon" && t != "log energy" && t != "logenergy"
               && t != "threshold" && t != "sure") {
        throw Error(keel::format("wentropy: unknown type '{}' (shannon / log energy / threshold / sure / norm)", type),
                    0, 0, "wentropy", "", "numkit:wentropy:type");
    }
    return t;
}

keel::String ddencmp_argcheck(keel::StringRef opt, keel::StringRef type)
{
    auto lower = [](keel::StringRef sv) {
        keel::String s(sv);
        s.to_ascii_lower_in_place();
        return s;
    };
    keel::String o = lower(opt);
    const keel::String t = lower(type);

    if (t == "wp")
        throw Error("ddencmp: wavelet-packet ('wp') defaults are not yet "
                    "supported (use 'wv')",
                    0, 0, "ddencmp", "", "numkit:ddencmp:wp");
    if (t != "wv")
        throw Error("ddencmp: type must be 'wv' (wavelet) or 'wp'",
                    0, 0, "ddencmp", "", "numkit:ddencmp:type");
    if (o != "den" && o != "cmp")
        throw Error("ddencmp: option must be 'den' (denoise) or 'cmp' (compress)",
                    0, 0, "ddencmp", "", "numkit:ddencmp:opt");
    return o;
}

} // anonymous

Value wthresh(const Value &X, keel::StringRef sorh, const Value &T,
              keel::Allocator *mr)
{
    wthresh_argcheck(X, sorh, T);
    const bool hard = (sorh.equals_ignore_ascii_case("h"));
    const bool elemT = T.numel() != 1;

    const keel::usize N = X.numel();
    const bool wantSingle =
        (X.type() == ValueType::SINGLE || X.isSingleComplex() ||
         T.type() == ValueType::SINGLE || T.isSingleComplex());
    const ValueType outT = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    const bool cplx = X.type() == ValueType::COMPLEX;
    Value Y = cplx ? Value::complexMatrix(X.dims().rows(), X.dims().cols(), mr)
                   : Value::matrix(X.dims().rows(), X.dims().cols(), outT, mr);
    if (cplx && wantSingle)
        Y.setSingleComplex(true);
    if (N == 0) return Y;

    auto tOf = [&](keel::usize i) {
        return elemT ? T.elemAsDouble(i) : T.elemAsDouble(0);
    };

    if (cplx) {
        keel::c64 *yd = Y.complexDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::c64 z = X.complexElem(i);
            const keel::f64 az = keel::abs(z);
            const keel::f64 T_i = tOf(i);
            if (hard) {
                yd[i] = (az > T_i) ? z : keel::c64(0.0, 0.0);
            } else {
                const keel::f64 mag = az - T_i;
                yd[i] = (mag > 0.0 && az > 0.0)
                            ? z * (mag / az)
                            : keel::c64(0.0, 0.0);
            }
        }
        return Y;
    }

    keel::f64 *yd = outT == ValueType::SINGLE
        ? nullptr
        : Y.doubleDataMut();
    keel::f32 *yf = outT == ValueType::SINGLE ? Y.singleDataMut() : nullptr;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 x = X.elemAsDouble(i);
        const keel::f64 ax = keel::abs(x);
        const keel::f64 T_i = tOf(i);
        keel::f64 r;
        if (hard) {
            r = (ax > T_i) ? x : 0.0;
        } else {
            // soft: sign(x) * max(|x| - T, 0)
            const keel::f64 mag = ax - T_i;
            r = (mag > 0.0) ? std::copysign(mag, x) : 0.0;
        }
        if (yf) yf[i] = keel::as<keel::f32>(r);
        else    yd[i] = r;
    }
    return Y;
}

Value wnoisest(const Value &C, const Value &L, const Value &S, keel::Allocator *mr)
{
    const keel::usize k = S.numel();
    Value out = Value::matrix(1, k, ValueType::DOUBLE, mr);
    if (k == 0) return out;
    keel::f64 *od = out.doubleDataMut();

    for (keel::usize i = 0; i < k; ++i) {
        const keel::i32 level = keel::as<keel::i32>(S.elemAsDouble(i));
        Value cD = detcoef(C, L, level, mr);
        auto v = vecFromValue(cD, mr);
        od[i] = keel::wavelet::noise_sigma_mad(keel::ArrayRef<keel::f64>(v.data(), v.size()), mr);
    }
    return out;
}

Value wdenoise(const Value &x, keel::i32 level, keel::StringRef wname, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    if (N < 2) {
        Value y = Value::matrix(x.dims().rows(), x.dims().cols(),
                                ValueType::DOUBLE, mr);
        keel::f64 *yd = y.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) yd[i] = x.elemAsDouble(i);
        return y;
    }
    const keel::StringRef w = wname.empty() ? keel::StringRef("sym4") : wname;
    auto fb = wavelet_filters(w);
    keel::wavelet::FilterBank kfb{std::move(fb.Lo_D), std::move(fb.Hi_D),
                                  std::move(fb.Lo_R), std::move(fb.Hi_R)};
    auto xv = vecFromValue(x, mr);
    auto rec = keel::wavelet::denoise_visushrink(keel::ArrayRef<keel::f64>(xv.data(), xv.size()),
                                                 keel::as<keel::i32>(level), kfb, mr);
    Value out = Value::matrix(1, rec.size(), ValueType::DOUBLE, mr);
    if (!rec.empty()) std::copy(rec.begin(), rec.end(), out.doubleDataMut());
    return out;
}

Value wentropy(const Value &X, keel::StringRef type, keel::f64 param,
               keel::Allocator *mr)
{
    const keel::String t = wentropy_argcheck(type, param);
    auto x = vecFromValue(X, mr);
    keel::wavelet::EntropyKind kind = keel::wavelet::EntropyKind::Norm;
    if (t == "shannon") kind = keel::wavelet::EntropyKind::Shannon;
    else if (t == "log energy" || t == "logenergy") kind = keel::wavelet::EntropyKind::LogEnergy;
    else if (t == "threshold") kind = keel::wavelet::EntropyKind::Threshold;
    else if (t == "sure") kind = keel::wavelet::EntropyKind::Sure;
    const keel::f64 e = keel::wavelet::entropy(keel::ArrayRef<keel::f64>(x.data(), x.size()), kind, param);
    return Value::scalar(e, mr);
}

DdencmpResult ddencmp(keel::StringRef opt, keel::StringRef type,
                      const Value &x, keel::Allocator *mr)
{
    const keel::String o = ddencmp_argcheck(opt, type);

    // Noise level from the finest-detail coefficients of a 1-level db1 DWT.
    auto [cA, cD] = dwt(x, "db1", mr);
    (void)cA;
    const keel::f64 med = median_abs(vecFromValue(cD));   // median(|cD1|)
    const keel::f64 sigma = med / 0.6745;
    const keel::f64 n = keel::as<keel::f64>(x.numel());

    DdencmpResult r;
    if (o == "den") {
        r.thr = Value::scalar(std::sqrt(2.0 * std::log(n)) * sigma, mr);
        r.sorh = Value::fromString("s", mr);
    } else { // "cmp"
        r.thr = Value::scalar(med, mr);
        r.sorh = Value::fromString("h", mr);
    }
    r.keepapp = Value::scalar(1.0, mr);
    return r;
}

} // namespace numkit::wavelet
