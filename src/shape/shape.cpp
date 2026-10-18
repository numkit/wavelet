// toolboxes/wavelet/src/shape/shape.cpp
//
// Continuous wavelet shape primitives: mexihat, morlet, meyeraux.

#include <numkit/wavelet/shape/shape.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>
#include <numkit/signal/transforms/fft.hpp>


// Compute-only TU: Value substrate + Error, no engine. The mexihat /
// morlet / meyeraux / shanwavf / cmorwavf / fbspwavf / centfrq / scal2frq
// builtins (CallContext wrappers) live in shape/shape_reg.cpp.
#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::wavelet {

namespace {

// linspace(LB, UB, N) into a freshly-allocated Value row.
Value linspace_row(double lb, double ub, size_t N, std::pmr::memory_resource *mr)
{
    Value xv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    if (N == 0) return xv;
    double *xd = xv.doubleDataMut();
    if (N == 1) { xd[0] = ub; return xv; }
    const double step = (ub - lb) / static_cast<double>(N - 1);
    for (size_t i = 0; i < N; ++i)
        xd[i] = lb + step * static_cast<double>(i);
    xd[N - 1] = ub;  // exact endpoint
    return xv;
}

} // anonymous

std::tuple<Value, Value>
mexihat(double lb, double ub, size_t N, std::pmr::memory_resource *mr)
{
    Value xv = linspace_row(lb, ub, N, mr);
    Value pv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    const double *xd = xv.doubleData();
    double *pd = pv.doubleDataMut();
    // C = 2/√3 · π^(-1/4)
    const double C = (2.0 / std::sqrt(3.0)) * std::pow(M_PI, -0.25);
    for (size_t i = 0; i < N; ++i) {
        const double t = xd[i];
        const double t2 = t * t;
        pd[i] = C * (1.0 - t2) * std::exp(-t2 / 2.0);
    }
    return {std::move(pv), std::move(xv)};
}

std::tuple<Value, Value>
morlet(double lb, double ub, size_t N, std::pmr::memory_resource *mr)
{
    Value xv = linspace_row(lb, ub, N, mr);
    Value pv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    const double *xd = xv.doubleData();
    double *pd = pv.doubleDataMut();
    for (size_t i = 0; i < N; ++i) {
        const double t = xd[i];
        pd[i] = std::exp(-t * t / 2.0) * std::cos(5.0 * t);
    }
    return {std::move(pv), std::move(xv)};
}

namespace {
// Normalized sinc: sin(π x) / (π x), with sinc(0) = 1.
inline double sinc_pi(double x)
{
    if (x == 0.0) return 1.0;
    const double a = M_PI * x;
    return std::sin(a) / a;
}

// Complex exponential exp(2πi·k·x) → cos(2πk x) + i·sin(2πk x).
inline Complex cexp_2pi(double kx)
{
    const double a = 2.0 * M_PI * kx;
    return {std::cos(a), std::sin(a)};
}

} // anonymous

std::tuple<Value, Value>
shanwavf(double lb, double ub, size_t N, double fb, double fc, std::pmr::memory_resource *mr)
{
    Value xv = linspace_row(lb, ub, N, mr);
    Value pv = Value::matrix(1, N, ValueType::COMPLEX, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    const double *xd = xv.doubleData();
    Complex *pd = pv.complexDataMut();
    const double s = std::sqrt(fb);
    for (size_t i = 0; i < N; ++i) {
        const double t = xd[i];
        const Complex e = cexp_2pi(fc * t);
        pd[i] = s * sinc_pi(fb * t) * e;
    }
    return {std::move(pv), std::move(xv)};
}

std::tuple<Value, Value>
cmorwavf(double lb, double ub, size_t N, double fb, double fc, std::pmr::memory_resource *mr)
{
    Value xv = linspace_row(lb, ub, N, mr);
    Value pv = Value::matrix(1, N, ValueType::COMPLEX, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    const double *xd = xv.doubleData();
    Complex *pd = pv.complexDataMut();
    const double inv = 1.0 / std::sqrt(M_PI * fb);
    for (size_t i = 0; i < N; ++i) {
        const double t = xd[i];
        const Complex e = cexp_2pi(fc * t);
        pd[i] = inv * std::exp(-t * t / fb) * e;
    }
    return {std::move(pv), std::move(xv)};
}

std::tuple<Value, Value>
fbspwavf(double lb, double ub, size_t N, int m, double fb, double fc, std::pmr::memory_resource *mr)
{
    if (m < 1)
        throw Error("fbspwavf: order m must be >= 1",
                    0, 0, "fbspwavf", "", "numkit:fbspwavf:order");
    Value xv = linspace_row(lb, ub, N, mr);
    Value pv = Value::matrix(1, N, ValueType::COMPLEX, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};
    const double *xd = xv.doubleData();
    Complex *pd = pv.complexDataMut();
    const double s = std::sqrt(fb);
    const double inv_m = 1.0 / static_cast<double>(m);
    for (size_t i = 0; i < N; ++i) {
        const double t = xd[i];
        const double sc = sinc_pi(fb * t * inv_m);
        const double pow_sc = std::pow(sc, double(m));
        const Complex e = cexp_2pi(fc * t);
        pd[i] = s * pow_sc * e;
    }
    return {std::move(pv), std::move(xv)};
}

Value meyeraux(const Value &x, std::pmr::memory_resource *mr)
{
    internal::requireNonempty(x, "meyeraux", "X", 1);
    internal::requireWaveClass(x, "meyeraux", "X", 1,
                               internal::WvClasses::FloatOnly);
    const auto poly = [](double v) {
        if (v <= 0.0) return 0.0;
        if (v >= 1.0) return 1.0;
        const double v2 = v * v;
        const double v3 = v2 * v;
        const double v4 = v2 * v2;
        return v4 * (35.0 - 84.0 * v + 70.0 * v2 - 20.0 * v3);
    };
    if (x.numel() == 1) {
        const double r = poly(x.elemAsDouble(0));
        if (x.type() == ValueType::SINGLE) {
            Value out = Value::matrix(1, 1, ValueType::SINGLE, mr);
            out.singleDataMut()[0] = static_cast<float>(r);
            return out;
        }
        return Value::scalar(r, mr);
    }
    Value out = internal::allocLike(x, x.type(), mr);
    for (std::size_t i = 0; i < x.numel(); ++i)
        internal::storeElem(out, i, poly(x.elemAsDouble(i)));
    return out;
}

namespace {

void centfrq_argcheck(const std::string &wname, int iter)
{
    if (wname.empty())
        throw Error("centfrq: wavelet name must be a non-empty string",
                    0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
    if (iter < 1 || iter > 20)
        throw Error("centfrq: ITER must be a positive integer in 1..20 (got " +
                    std::to_string(iter) + ")",
                    0, 0, "centfrq", "", "numkit:centfrq:invalidIter");
}

void scal2frq_argcheck(const Value &a, const std::string &wname, double delta)
{
    internal::requireNonempty(a, "scal2frq", "A", 1);
    if (a.type() != ValueType::DOUBLE && a.type() != ValueType::SINGLE)
        throw Error("scal2frq: A must be a real double or single vector",
                    0, 0, "scal2frq", "", "numkit:scal2frq:invalidScales");
    const auto &dm = a.dims();
    if (dm.ndims() > 2 || (dm.rows() != 1 && dm.cols() != 1))
        throw Error("scal2frq: A must be a 1-D vector of positive scales",
                    0, 0, "scal2frq", "", "numkit:scal2frq:invalidScales");
    for (size_t i = 0; i < a.numel(); ++i) {
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


std::vector<double> conv_full_vec(const std::vector<double> &a,
                                  const std::vector<double> &b)
{
    if (a.empty() || b.empty()) return {};
    std::vector<double> c(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            c[i + j] += a[i] * b[j];
    return c;
}

bool parse_two_floats(const std::string &s, double &v1, double &v2)
{
    const size_t dash = s.find('-');
    if (dash == std::string::npos || dash == 0 || dash + 1 >= s.size())
        return false;
    char *e1 = nullptr;
    char *e2 = nullptr;
    const std::string p1 = s.substr(0, dash);
    const std::string p2 = s.substr(dash + 1);
    v1 = std::strtod(p1.c_str(), &e1);
    v2 = std::strtod(p2.c_str(), &e2);
    return (e1 && *e1 == '\0' && e2 && *e2 == '\0' && v1 > 0.0 && v2 > 0.0);
}

bool parse_int_two_floats(const std::string &s, int &m, double &v1, double &v2)
{
    const size_t d1 = s.find('-');
    if (d1 == std::string::npos || d1 == 0 || d1 + 1 >= s.size())
        return false;
    char *em = nullptr;
    const std::string pm = s.substr(0, d1);
    const long mv = std::strtol(pm.c_str(), &em, 10);
    if (!em || *em != '\0' || mv < 1)
        return false;
    m = static_cast<int>(mv);
    return parse_two_floats(s.substr(d1 + 1), v1, v2);
}

std::tuple<Value, Value> build_wavefun_psi(const std::string &raw_name,
                                           int iter,
                                           std::pmr::memory_resource *mr)
{
    std::string wname = raw_name;
    for (char &c : wname)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    const size_t gridN = static_cast<size_t>(1ULL << iter);

    if (wname == "morl")
        return morlet(-8.0, 8.0, gridN, mr);
    if (wname == "mexh")
        return mexihat(-8.0, 8.0, gridN, mr);

    if (wname.rfind("gaus", 0) == 0) {
        int p = 1;
        if (wname.size() > 4) {
            char *ep = nullptr;
            const long pv = std::strtol(wname.c_str() + 4, &ep, 10);
            if (!ep || *ep != '\0' || pv < 1 || pv > 8)
                throw Error("centfrq: invalid Gaussian wavelet order in '" + raw_name + "'",
                            0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
            p = static_cast<int>(pv);
        }
        return gauswavf(-5.0, 5.0, gridN, p, mr);
    }

    if (wname.rfind("cgau", 0) == 0) {
        int p = 1;
        if (wname.size() > 4) {
            char *ep = nullptr;
            const long pv = std::strtol(wname.c_str() + 4, &ep, 10);
            if (!ep || *ep != '\0' || pv < 1 || pv > 8)
                throw Error("centfrq: invalid complex Gaussian wavelet order in '" + raw_name + "'",
                            0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
            p = static_cast<int>(pv);
        }
        return cgauwavf(-5.0, 5.0, gridN, p, mr);
    }

    if (wname.rfind("shan", 0) == 0) {
        double fb = 1.0, fc = 1.5;
        if (wname.size() > 4 && !parse_two_floats(wname.substr(4), fb, fc))
            throw Error("centfrq: invalid Shannon wavelet specification '" + raw_name + "'",
                        0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
        return shanwavf(-20.0, 20.0, gridN, fb, fc, mr);
    }

    if (wname.rfind("cmor", 0) == 0) {
        double fb = 1.0, fc = 1.5;
        if (wname.size() > 4 && !parse_two_floats(wname.substr(4), fb, fc))
            throw Error("centfrq: invalid complex Morlet wavelet specification '" + raw_name + "'",
                        0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
        return cmorwavf(-8.0, 8.0, gridN, fb, fc, mr);
    }

    if (wname.rfind("fbsp", 0) == 0) {
        int m = 1;
        double fb = 1.0, fc = 1.5;
        if (wname.size() > 4 && !parse_int_two_floats(wname.substr(4), m, fb, fc))
            throw Error("centfrq: invalid B-spline wavelet specification '" + raw_name + "'",
                        0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
        return fbspwavf(-20.0, 20.0, gridN, m, fb, fc, mr);
    }

    // Orthogonal / biorthogonal filter-bank wavelets via two-scale cascade.
    FilterBank fb = wavelet_filters(wname);
    const bool is_bior = (wname.rfind("bior", 0) == 0 || wname.rfind("rbio", 0) == 0);
    std::vector<double> Lo, Hi;
    if (is_bior) {
        // Decomposition wavelet psi1 uses time-reversed Lo_D and Hi_D.
        Lo.assign(fb.Lo_D.rbegin(), fb.Lo_D.rend());
        Hi.assign(fb.Hi_D.rbegin(), fb.Hi_D.rend());
    } else {
        Lo = fb.Lo_R;
        Hi = fb.Hi_R;
    }

    const bool negate_psi =
        (wname.rfind("sym", 0) == 0 || wname.rfind("coif", 0) == 0 ||
         wname.rfind("bior2.", 0) == 0 || wname.rfind("bior3.", 0) == 0 ||
         wname == "bior4.4" || wname.rfind("rbio2.", 0) == 0 ||
         wname.rfind("rbio3.", 0) == 0 || wname == "rbio4.4");
    if (negate_psi) {
        for (double &v : Hi) v = -v;
    }

    std::vector<double> y = Hi;
    for (int step = 1; step < iter; ++step) {
        std::vector<double> up(2 * y.size() - 1, 0.0);
        for (size_t k = 0; k < y.size(); ++k)
            up[2 * k] = y[k];
        y = conv_full_vec(up, Lo);
    }
    const double scale = std::pow(2.0, 0.5 * static_cast<double>(iter));
    for (double &v : y) v *= scale;

    const size_t L = Lo.size();
    size_t len_psi = 0;
    double T = 0.0;
    if (L == 2) {
        len_psi = gridN + 2;
        T = 1.0 + 1.0 / static_cast<double>(gridN);
    } else if (is_bior) {
        len_psi = (L - 1) * gridN;
        T = static_cast<double>(L - 1) - 1.0 / static_cast<double>(gridN);
    } else {
        len_psi = (L - 1) * gridN + 1;
        T = static_cast<double>(L - 1);
    }

    Value psi = Value::matrix(1, len_psi, ValueType::DOUBLE, mr);
    double *pd = psi.doubleDataMut();
    std::fill(pd, pd + len_psi, 0.0);
    const size_t copy_len = std::min(y.size(), len_psi > 1 ? len_psi - 1 : 0);
    std::copy(y.begin(), y.begin() + copy_len, pd + 1);

    Value xval = linspace_row(0.0, T, len_psi, mr);
    return {std::move(psi), std::move(xval)};
}

} // anonymous

CentfrqResult centfrqGrid(const std::string &wname, int iter,
                          bool withRecfreq,
                          std::pmr::memory_resource *mr)
{
    centfrq_argcheck(wname, iter);
    auto [psi, xval] = build_wavefun_psi(wname, iter, mr);
    const size_t n = psi.numel();
    Value Y = signal::fft(psi, -1, 0, mr);


    auto getComplex = [](const Value &v, size_t idx) -> Complex {
        if (v.type() == ValueType::COMPLEX)
            return v.complexData()[idx];
        return Complex(v.elemAsDouble(idx), 0.0);
    };

    size_t best_idx0 = 0;
    double best_mag = -1.0;
    for (size_t i = 0; i < n; ++i) {
        const Complex c = getComplex(Y, i);
        const double mag = std::hypot(c.real(), c.imag());
        if (mag > best_mag) {
            best_mag = mag;
            best_idx0 = i;
        }
    }

    const size_t ind = best_idx0 + 1; // 1-based index
    const size_t ind_fold =
        (static_cast<double>(ind) > 0.5 * static_cast<double>(n))
            ? (n + 2 - ind)
            : ind;
    const double T = xval.elemAsDouble(n - 1) - xval.elemAsDouble(0);
    const double freq = static_cast<double>(ind_fold - 1) / T;

    Value recfreq = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (withRecfreq) {
        const Complex Ypeak = getComplex(Y, best_idx0);

        const double inv_n = 1.0 / static_cast<double>(n);
        if (psi.isComplex()) {
            recfreq = Value::matrix(1, n, ValueType::COMPLEX, mr);
            Complex *rd = recfreq.complexDataMut();
            for (size_t k = 0; k < n; ++k) {
                const double ang = 2.0 * M_PI * static_cast<double>(best_idx0 * k) * inv_n;
                const Complex e(std::cos(ang), std::sin(ang));
                rd[k] = (Ypeak * e) * inv_n;
            }
        } else {
            recfreq = Value::matrix(1, n, ValueType::DOUBLE, mr);
            double *rd = recfreq.doubleDataMut();
            const double scale = (best_idx0 == 0 || (n % 2 == 0 && best_idx0 == n / 2))
                                     ? inv_n
                                     : 2.0 * inv_n;
            for (size_t k = 0; k < n; ++k) {
                const double ang = 2.0 * M_PI * static_cast<double>(best_idx0 * k) * inv_n;
                const Complex e(std::cos(ang), std::sin(ang));
                rd[k] = (Ypeak * e).real() * scale;
            }
        }
    }

    return {freq, std::move(xval), std::move(recfreq)};
}

double centfrq(const std::string &wname, int iter)
{
    return centfrqGrid(wname, iter, false, nullptr).freq;
}

Value scal2frq(const Value &a, const std::string &wname,
               double delta, std::pmr::memory_resource *mr)
{
    scal2frq_argcheck(a, wname, delta);
    const double cf = centfrq(wname, 8);
    const double num = cf / delta;
    Value out = internal::allocLike(a, a.type(), mr);
    for (size_t i = 0; i < a.numel(); ++i)
        internal::storeElem(out, i, num / a.elemAsDouble(i));
    return out;
}

namespace {

void cwt_argcheck(const Value &x, const std::string &wname, double fs,
                  int voicesPerOctave, double timeBandwidth, double gamma,
                  const Value &freqLimits)
{
    if (!x.isNumeric() || x.isComplex())
        throw Error("cwt: X must be a real numeric vector",
                    0, 0, "cwt", "", "numkit:cwt:expectedReal");
    if (x.dims().ndims() > 2 || (x.dims().rows() > 1 && x.dims().cols() > 1) || x.numel() < 4)
        throw Error("cwt: X must be a 1-D vector with at least 4 samples",
                    0, 0, "cwt", "", "numkit:cwt:expectedVector");
    for (size_t i = 0; i < x.numel(); ++i) {
        if (!std::isfinite(x.elemAsDouble(i)))
            throw Error("cwt: X must be finite",
                        0, 0, "cwt", "", "numkit:cwt:expectedFinite");
    }
    std::string wl = wname;
    for (char &c : wl)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (wl != "morse" && wl != "amor" && wl != "bump")
        throw Error("cwt: WAVELET must be 'morse', 'amor', or 'bump' (got '" + wname + "')",
                    0, 0, "cwt", "", "numkit:cwt:invalidWavelet");
    if (!(fs > 0.0) || !std::isfinite(fs))
        throw Error("cwt: FS must be a positive finite scalar",
                    0, 0, "cwt", "", "numkit:cwt:invalidFs");
    if (voicesPerOctave < 1 || voicesPerOctave > 48)
        throw Error("cwt: VoicesPerOctave must be an integer in [1, 48]",
                    0, 0, "cwt", "", "numkit:cwt:invalidVoicesPerOctave");
    if (wl == "morse") {
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
}

} // anonymous

CwtResult cwt(const Value &x,
              const std::string &wname,
              double fs,
              int voicesPerOctave,
              double timeBandwidth,
              double gamma,
              bool extendSignal,
              const Value &freqLimits,
              std::pmr::memory_resource *mr)
{
    cwt_argcheck(x, wname, fs, voicesPerOctave, timeBandwidth, gamma, freqLimits);

    std::string wl = wname;
    for (char &c : wl)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    const size_t N = x.numel();
    const double nv = static_cast<double>(voicesPerOctave);
    const double beta = timeBandwidth / gamma;

    double wpsi = 0.0;
    double wc = 0.0;
    double sigt = 0.0;

    if (wl == "morse") {
        wpsi = std::pow(beta / gamma, 1.0 / gamma);
        const double target = std::log(0.5);
        const double wpsi_ga = beta / gamma;
        double w = wpsi * (1.0 + std::sqrt(2.0 * std::log(2.0) / (beta * gamma)));
        for (int it = 0; it < 20; ++it) {
            const double w_ga = std::pow(w, gamma);
            const double g = beta * std::log(w / wpsi) - (w_ga - wpsi_ga) - target;
            const double gp = beta / w - gamma * (w_ga / w);
            if (std::abs(gp) < 1e-30) break;
            const double step = g / gp;
            w -= step;
            if (w <= wpsi) w = 0.5 * (w + step + wpsi);
            if (std::abs(step) <= 1e-15 * w) break;
        }
        wc = w;

        auto log_term = [&](double a) {
            return std::lgamma(a / gamma) - (a / gamma) * std::log(2.0);
        };
        const double ld = log_term(2.0 * beta + 1.0);
        const double t1 = (beta * beta) * std::exp(log_term(2.0 * beta - 1.0) - ld);
        const double t2 = (gamma * gamma) * std::exp(log_term(2.0 * beta + 2.0 * gamma - 1.0) - ld);
        const double t3 = (2.0 * beta * gamma) * std::exp(log_term(2.0 * beta + gamma - 1.0) - ld);
        sigt = wpsi * std::sqrt(std::max(0.0, t1 + t2 - t3));
    } else if (wl == "amor") {
        wpsi = 6.0;
        wc = wpsi + std::sqrt(2.0 * std::log(10.0));
        sigt = std::sqrt(2.0);
    } else {
        // bump: mu = 5, sigma = 0.6
        wpsi = 5.0;
        const double sig = 0.6;
        wc = wpsi + sig * std::sqrt(1.0 - 1.0 / (1.0 - std::log(0.1)));
        sigt = 5.847705277601328;
    }

    double s0 = wc / M_PI;
    double smax = static_cast<double>(N) / (2.0 * sigt);
    if (smax < s0 * std::pow(2.0, 1.0 / nv))
        smax = s0 * std::pow(2.0, 1.0 / nv);

    if (!freqLimits.isEmpty()) {
        const double fmin = freqLimits.elemAsDouble(0) / fs;
        const double fmax = freqLimits.elemAsDouble(1) / fs;
        const double s_from_fmax = wpsi / (2.0 * M_PI * fmax);
        const double s_from_fmin = wpsi / (2.0 * M_PI * fmin);
        if (s_from_fmax > s0) s0 = s_from_fmax;
        if (s_from_fmin < smax) smax = s_from_fmin;
        if (smax < s0) smax = s0;
    }

    const double nOct = std::max(0.0, std::log2(smax / s0));
    const size_t nScales = 1 + static_cast<size_t>(std::floor(nv * nOct + 1e-12));

    std::vector<double> scales(nScales);
    std::vector<double> freqs(nScales);
    for (size_t j = 0; j < nScales; ++j) {
        scales[j] = s0 * std::pow(2.0, static_cast<double>(j) / nv);
        freqs[j] = (wpsi * fs) / (2.0 * M_PI * scales[j]);
    }

    const size_t padL = extendSignal ? std::min<size_t>(N / 2, 100000) : 0;
    const size_t Npad = N + 2 * padL;

    Value xpad = Value::matrix(1, Npad, ValueType::DOUBLE, mr);
    double *xpd = xpad.doubleDataMut();
    for (size_t i = 0; i < padL; ++i)
        xpd[i] = x.elemAsDouble(padL - 1 - i);
    for (size_t i = 0; i < N; ++i)
        xpd[padL + i] = x.elemAsDouble(i);
    for (size_t i = 0; i < padL; ++i)
        xpd[padL + N + i] = x.elemAsDouble(N - 1 - i);

    Value X = signal::fft(xpad, -1, 0, mr);
    auto getX = [&](size_t k) -> Complex {
        if (X.type() == ValueType::COMPLEX)
            return X.complexData()[k];
        return Complex(X.elemAsDouble(k), 0.0);
    };

    const size_t kMax = Npad / 2;
    std::vector<double> omega(kMax + 1);
    const double dOmega = (2.0 * M_PI) / static_cast<double>(Npad);
    for (size_t k = 0; k <= kMax; ++k)
        omega[k] = static_cast<double>(k) * dOmega;

    const bool outSingle = (x.type() == ValueType::SINGLE);
    Value cfs = Value::matrix(nScales, N, ValueType::COMPLEX, mr);
    Complex *cfs_d = cfs.complexDataMut();

    Value prodRow = Value::matrix(1, Npad, ValueType::COMPLEX, mr);
    Complex *prd = prodRow.complexDataMut();
    const double wpsi_ga = (wl == "morse") ? (beta / gamma) : 0.0;

    for (size_t j = 0; j < nScales; ++j) {
        const double s = scales[j];
        std::fill(prd, prd + Npad, Complex(0.0, 0.0));
        for (size_t k = 1; k <= kMax; ++k) {
            const double sw = s * omega[k];
            double Hval = 0.0;
            if (wl == "morse") {
                if (sw > 0.0) {
                    const double log_H = beta * std::log(sw / wpsi) - (std::pow(sw, gamma) - wpsi_ga);
                    if (log_H > -745.0)
                        Hval = 2.0 * std::exp(log_H);
                }
            } else if (wl == "amor") {
                const double diff = sw - wpsi;
                Hval = 2.0 * std::exp(-0.5 * diff * diff);
            } else {
                const double u = (sw - wpsi) / 0.6;
                if (std::abs(u) < 1.0 - 1e-15)
                    Hval = 2.0 * std::exp(1.0 - 1.0 / (1.0 - u * u));
            }
            if (Hval != 0.0)
                prd[k] = getX(k) * Hval;
        }

        Value yRow = signal::ifft(prodRow, -1, 0, mr);
        for (size_t n = 0; n < N; ++n) {
            const Complex c = (yRow.type() == ValueType::COMPLEX)
                                  ? yRow.complexData()[padL + n]
                                  : Complex(yRow.elemAsDouble(padL + n), 0.0);
            const size_t idx = j + n * nScales; // column-major (nScales x N)
            cfs_d[idx] = c;
        }
    }

    Value fOut = Value::matrix(nScales, 1,
                               outSingle ? ValueType::SINGLE : ValueType::DOUBLE,
                               mr);
    for (size_t j = 0; j < nScales; ++j)
        internal::storeElem(fOut, j, freqs[j]);

    Value coiOut = Value::matrix(N, 1,
                                 outSingle ? ValueType::SINGLE : ValueType::DOUBLE,
                                 mr);
    const double coiNum = (fs * wpsi * sigt) / (2.0 * M_PI);
    const double fTop = freqs[0];
    for (size_t n = 0; n < N; ++n) {
        const size_t dist = std::min(n + 1, N - n);
        const double val = std::min(fTop, coiNum / static_cast<double>(dist));
        internal::storeElem(coiOut, n, val);
    }

    return {std::move(cfs), std::move(fOut), std::move(coiOut)};
}

} // namespace numkit::wavelet

