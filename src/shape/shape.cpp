// toolboxes/wavelet/src/shape/shape.cpp
// Continuous wavelet shapes, centfrq, scal2frq, and cwt — delegated to keel::wavelet.

#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <keel/wavelet.h>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>
#include <numkit/wavelet/shape/shape.hpp>

#include <cmath>
#include <cstring>

namespace numkit::wavelet {

namespace {

void shanwavf_argcheck(double /*lb*/, double /*ub*/, keel::usize /*N*/, double fb, double fc)
{
    if (!(fb > 0.0) || !std::isfinite(fb))
        throw Error("shanwavf: bandwidth FB must be positive and finite",
                    0, 0, "shanwavf", "", "numkit:shanwavf:fb");
    if (!(fc > 0.0) || !std::isfinite(fc))
        throw Error("shanwavf: center frequency FC must be positive and finite",
                    0, 0, "shanwavf", "", "numkit:shanwavf:fc");
}

void cmorwavf_argcheck(double /*lb*/, double /*ub*/, keel::usize /*N*/, double fb, double fc)
{
    if (!(fb > 0.0) || !std::isfinite(fb))
        throw Error("cmorwavf: bandwidth FB must be positive and finite",
                    0, 0, "cmorwavf", "", "numkit:cmorwavf:fb");
    if (!(fc > 0.0) || !std::isfinite(fc))
        throw Error("cmorwavf: center frequency FC must be positive and finite",
                    0, 0, "cmorwavf", "", "numkit:cmorwavf:fc");
}

void fbspwavf_argcheck(double /*lb*/, double /*ub*/, keel::usize /*N*/, int m, double fb, double fc)
{
    if (m < 1)
        throw Error("fbspwavf: order m must be >= 1",
                    0, 0, "fbspwavf", "", "numkit:fbspwavf:order");
    if (!(fb > 0.0) || !std::isfinite(fb))
        throw Error("fbspwavf: bandwidth FB must be positive and finite",
                    0, 0, "fbspwavf", "", "numkit:fbspwavf:fb");
    if (!(fc > 0.0) || !std::isfinite(fc))
        throw Error("fbspwavf: center frequency FC must be positive and finite",
                    0, 0, "fbspwavf", "", "numkit:fbspwavf:fc");
}

void meyeraux_argcheck(const Value &x)
{
    internal::requireNonempty(x, "meyeraux", "X", 1);
    internal::requireWaveClass(x, "meyeraux", "X", 1,
                               internal::WvClasses::FloatOnly);
}

void centfrq_argcheck(keel::StringRef wname, int iter)
{
    if (wname.empty())
        throw Error("centfrq: wavelet name must be a non-empty string",
                    0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
    if (iter < 1 || iter > 20)
        throw Error(keel::format("centfrq: ITER must be a positive integer in 1..20 (got {})",
                                 iter),
                    0, 0, "centfrq", "", "numkit:centfrq:invalidIter");
}

void scal2frq_argcheck(const Value &a, keel::StringRef wname, double delta)
{
    internal::requireNonempty(a, "scal2frq", "A", 1);
    if (a.type() != ValueType::DOUBLE && a.type() != ValueType::SINGLE)
        throw Error("scal2frq: A must be a real double or single vector",
                    0, 0, "scal2frq", "", "numkit:scal2frq:invalidScales");
    const auto &dm = a.dims();
    if (dm.ndims() > 2 || (dm.rows() != 1 && dm.cols() != 1))
        throw Error("scal2frq: A must be a 1-D vector of positive scales",
                    0, 0, "scal2frq", "", "numkit:scal2frq:invalidScales");
    for (keel::usize i = 0; i < a.numel(); ++i) {
        const double v = a.elemAsDouble(i);
        if (!std::isfinite(v) || v <= 0.0)
            throw Error("scal2frq: all scales in A must be finite and positive",
                        0, 0, "scal2frq", "", "numkit:scal2frq:invalidScales");
    }
    if (wname.empty())
        throw Error("scal2frq: wavelet name must be a non-empty string",
                    0, 0, "scal2frq", "", "numkit:scal2frq:invalidWavelet");
    if (!std::isfinite(delta) || delta <= 0.0)
        throw Error("scal2frq: sampling period DELTA must be finite and positive",
                    0, 0, "scal2frq", "", "numkit:scal2frq:invalidDelta");
}

keel::wavelet::CwtWavelet cwt_argcheck(const Value &x, keel::StringRef wname, double fs,
                                       int voicesPerOctave, double timeBandwidth, double gamma,
                                       const Value &freqLimits)
{
    if (!x.isNumeric() || x.isComplex())
        throw Error("cwt: X must be a real numeric vector",
                    0, 0, "cwt", "", "numkit:cwt:expectedReal");
    if (x.dims().ndims() > 2 || (x.dims().rows() > 1 && x.dims().cols() > 1) || x.numel() < 4)
        throw Error("cwt: X must be a 1-D vector with at least 4 samples",
                    0, 0, "cwt", "", "numkit:cwt:expectedVector");
    for (keel::usize i = 0; i < x.numel(); ++i) {
        if (!std::isfinite(x.elemAsDouble(i)))
            throw Error("cwt: X must be finite",
                        0, 0, "cwt", "", "numkit:cwt:expectedFinite");
    }
    keel::String wl(wname);
    for (char &c : wl)
        c = keel::to_ascii_lower(c);
    keel::wavelet::CwtWavelet wkind = keel::wavelet::CwtWavelet::Morse;
    if (wl == "morse") wkind = keel::wavelet::CwtWavelet::Morse;
    else if (wl == "amor") wkind = keel::wavelet::CwtWavelet::Amor;
    else if (wl == "bump") wkind = keel::wavelet::CwtWavelet::Bump;
    else
        throw Error(keel::format("cwt: WAVELET must be 'morse', 'amor', or 'bump' (got '{}')", wname),
                    0, 0, "cwt", "", "numkit:cwt:invalidWavelet");
    if (!(fs > 0.0) || !std::isfinite(fs))
        throw Error("cwt: FS must be a positive finite scalar",
                    0, 0, "cwt", "", "numkit:cwt:invalidFs");
    if (voicesPerOctave < 1 || voicesPerOctave > 48)
        throw Error("cwt: VoicesPerOctave must be an integer in [1, 48]",
                    0, 0, "cwt", "", "numkit:cwt:invalidVoicesPerOctave");
    if (wkind == keel::wavelet::CwtWavelet::Morse) {
        if (!(gamma > 0.0) || !std::isfinite(gamma))
            throw Error("cwt: gamma must be positive and finite",
                        0, 0, "cwt", "", "numkit:cwt:invalidGamma");
        if (!(timeBandwidth > gamma) || timeBandwidth > 40.0 * gamma || !std::isfinite(timeBandwidth))
            throw Error("cwt: TimeBandwidth must be in (gamma, 40*gamma]",
                        0, 0, "cwt", "", "numkit:cwt:invalidTimeBandwidth");
    }
    if (!freqLimits.isEmpty()) {
        if (!freqLimits.isNumeric() || freqLimits.isComplex() || freqLimits.numel() != 2)
            throw Error("cwt: FrequencyLimits must be a 2-element real vector [fmin, fmax]",
                        0, 0, "cwt", "", "numkit:cwt:invalidFrequencyLimits");
        const double fmin = freqLimits.elemAsDouble(0);
        const double fmax = freqLimits.elemAsDouble(1);
        if (!(fmin > 0.0) || !(fmax > fmin) || !std::isfinite(fmin) || !std::isfinite(fmax))
            throw Error("cwt: FrequencyLimits must satisfy 0 < fmin < fmax",
                        0, 0, "cwt", "", "numkit:cwt:invalidFrequencyLimits");
    }
    return wkind;
}

} // anonymous

std::tuple<Value, Value>
mexihat(double lb, double ub, keel::usize N, keel::Allocator *mr)
{
    Value xv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    Value pv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    keel::wavelet::mexihat(lb, ub,
                           keel::MutArrayRef<keel::f64>(pv.doubleDataMut(), N),
                           keel::MutArrayRef<keel::f64>(xv.doubleDataMut(), N));
    return {std::move(pv), std::move(xv)};
}

std::tuple<Value, Value>
morlet(double lb, double ub, keel::usize N, keel::Allocator *mr)
{
    Value xv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    Value pv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    keel::wavelet::morlet(lb, ub,
                          keel::MutArrayRef<keel::f64>(pv.doubleDataMut(), N),
                          keel::MutArrayRef<keel::f64>(xv.doubleDataMut(), N));
    return {std::move(pv), std::move(xv)};
}

std::tuple<Value, Value>
shanwavf(double lb, double ub, keel::usize N, double fb, double fc, keel::Allocator *mr)
{
    shanwavf_argcheck(lb, ub, N, fb, fc);
    Value xv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    Value pv = Value::matrix(1, N, ValueType::COMPLEX, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    keel::wavelet::shanwavf(lb, ub, fb, fc,
                            keel::MutArrayRef<keel::c64>(pv.complexDataMut(), N),
                            keel::MutArrayRef<keel::f64>(xv.doubleDataMut(), N));
    return {std::move(pv), std::move(xv)};
}

std::tuple<Value, Value>
cmorwavf(double lb, double ub, keel::usize N, double fb, double fc, keel::Allocator *mr)
{
    cmorwavf_argcheck(lb, ub, N, fb, fc);
    Value xv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    Value pv = Value::matrix(1, N, ValueType::COMPLEX, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    keel::wavelet::cmorwavf(lb, ub, fb, fc,
                            keel::MutArrayRef<keel::c64>(pv.complexDataMut(), N),
                            keel::MutArrayRef<keel::f64>(xv.doubleDataMut(), N));
    return {std::move(pv), std::move(xv)};
}

std::tuple<Value, Value>
fbspwavf(double lb, double ub, keel::usize N, int m, double fb, double fc, keel::Allocator *mr)
{
    fbspwavf_argcheck(lb, ub, N, m, fb, fc);
    Value xv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    Value pv = Value::matrix(1, N, ValueType::COMPLEX, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    keel::wavelet::fbspwavf(lb, ub, static_cast<keel::i32>(m), fb, fc,
                            keel::MutArrayRef<keel::c64>(pv.complexDataMut(), N),
                            keel::MutArrayRef<keel::f64>(xv.doubleDataMut(), N));
    return {std::move(pv), std::move(xv)};
}

Value meyeraux(const Value &x, keel::Allocator *mr)
{
    meyeraux_argcheck(x);
    if (x.numel() == 1) {
        const double r = keel::wavelet::meyeraux(x.elemAsDouble(0));
        if (x.type() == ValueType::SINGLE) {
            Value out = Value::matrix(1, 1, ValueType::SINGLE, mr);
            out.singleDataMut()[0] = static_cast<float>(r);
            return out;
        }
        return Value::scalar(r, mr);
    }
    Value out = internal::allocLike(x, x.type(), mr);
    for (keel::usize i = 0; i < x.numel(); ++i)
        internal::storeElem(out, i, keel::wavelet::meyeraux(x.elemAsDouble(i)));
    return out;
}

CentfrqResult centfrqGrid(keel::StringRef wname, int iter,
                          bool withRecfreq,
                          keel::Allocator *mr)
{
    centfrq_argcheck(wname, iter);
    auto r = keel::wavelet::centfrq(wname, static_cast<keel::i32>(iter), withRecfreq, mr);
    if (!r.ok) {
        throw Error(keel::format("centfrq: invalid or unsupported wavelet '{}'", wname),
                    0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
    }

    Value xval = Value::matrix(1, r.xval.size(), ValueType::DOUBLE, mr);
    if (!r.xval.empty()) std::memcpy(xval.doubleDataMut(), r.xval.data(), r.xval.size() * sizeof(double));

    Value recfreq = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (withRecfreq) {
        if (r.is_complex) {
            recfreq = Value::matrix(1, r.recfreq_cplx.size(), ValueType::COMPLEX, mr);
            if (!r.recfreq_cplx.empty())
                std::memcpy(recfreq.complexDataMut(), r.recfreq_cplx.data(),
                            r.recfreq_cplx.size() * sizeof(keel::c64));
        } else {
            recfreq = Value::matrix(1, r.recfreq_real.size(), ValueType::DOUBLE, mr);
            if (!r.recfreq_real.empty())
                std::memcpy(recfreq.doubleDataMut(), r.recfreq_real.data(),
                            r.recfreq_real.size() * sizeof(double));
        }
    }
    return {r.freq, std::move(xval), std::move(recfreq)};
}

double centfrq(keel::StringRef wname, int iter)
{
    return centfrqGrid(wname, iter, false, nullptr).freq;
}

Value scal2frq(const Value &a, keel::StringRef wname,
               double delta, keel::Allocator *mr)
{
    scal2frq_argcheck(a, wname, delta);
    const double cf = centfrq(wname, 8);
    const double num = cf / delta;
    Value out = internal::allocLike(a, a.type(), mr);
    for (keel::usize i = 0; i < a.numel(); ++i)
        internal::storeElem(out, i, num / a.elemAsDouble(i));
    return out;
}

CwtResult cwt(const Value &x,
              keel::StringRef wname,
              double fs,
              int voicesPerOctave,
              double timeBandwidth,
              double gamma,
              bool extendSignal,
              const Value &freqLimits,
              keel::Allocator *mr)
{
    const auto wkind = cwt_argcheck(x, wname, fs, voicesPerOctave, timeBandwidth, gamma, freqLimits);
    const keel::usize N = x.numel();
    keel::Array<keel::f64> xv(N, mr);
    for (keel::usize i = 0; i < N; ++i) xv[i] = x.elemAsDouble(i);

    const double fmin = freqLimits.isEmpty() ? 0.0 : freqLimits.elemAsDouble(0);
    const double fmax = freqLimits.isEmpty() ? 0.0 : freqLimits.elemAsDouble(1);

    auto r = keel::wavelet::cwt(xv, wkind, fs, static_cast<keel::i32>(voicesPerOctave),
                                timeBandwidth, gamma, extendSignal, fmin, fmax, mr);

    const bool outSingle = (x.type() == ValueType::SINGLE);
    Value cfs = Value::matrix(r.num_scales, r.num_samples, ValueType::COMPLEX, mr);
    if (!r.cfs_colmajor.empty())
        std::memcpy(cfs.complexDataMut(), r.cfs_colmajor.data(),
                    r.cfs_colmajor.size() * sizeof(keel::c64));

    Value fOut = Value::matrix(r.num_scales, 1,
                               outSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
    for (keel::usize j = 0; j < r.num_scales; ++j)
        internal::storeElem(fOut, j, r.freqs[j]);

    Value coiOut = Value::matrix(r.num_samples, 1,
                                 outSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
    for (keel::usize n = 0; n < r.num_samples; ++n)
        internal::storeElem(coiOut, n, r.coi[n]);

    return {std::move(cfs), std::move(fOut), std::move(coiOut)};
}

} // namespace numkit::wavelet
