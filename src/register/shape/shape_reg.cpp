// toolboxes/wavelet/src/shape/shape_reg.cpp
//
// Register half of the continuous-wavelet shape primitives: the CallContext
// builtins mexihat / morlet / meyeraux / shanwavf / cmorwavf / fbspwavf
// (argument parsing + multi-output psi/x packing) that delegate to the
// engine-free compute in shape.cpp. shape_grid_reg is the shared
// register-side dispatcher. library.cpp forward-declares + registers these.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/wavelet/shape/shape.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <cctype>
#include <string>
#include <tuple>
#include <utility>

namespace numkit::wavelet {
namespace detail {

static void shape_grid_reg(const char *fn,
                           std::tuple<Value, Value> (*impl)(double, double, size_t, std::pmr::memory_resource *),
                           Span<const Value> args, size_t nargout,
                           Span<Value> outs, CallContext &ctx)
{
    if (args.size() < 3)
        throw Error(std::string(fn) + ": requires (LB, UB, N)",
                    0, 0, fn, "", "numkit:wav:nargin");
    const double lb = args[0].toScalar();
    const double ub = args[1].toScalar();
    const double Nd = args[2].toScalar();
    if (!(Nd >= 0.0))
        throw Error(std::string(fn) + ": N must be non-negative",
                    0, 0, fn, "", "numkit:wav:N");
    const size_t N = static_cast<size_t>(Nd);
    auto [psi, x] = impl(lb, ub, N, ctx.engine->resource());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

void mexihat_reg(Span<const Value> args, size_t nargout,
                 Span<Value> outs, CallContext &ctx)
{
    shape_grid_reg("mexihat", &mexihat, args, nargout, outs, ctx);
}

void morlet_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    shape_grid_reg("morlet", &morlet, args, nargout, outs, ctx);
}

void meyeraux_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    if (args.empty())
        throw Error("meyeraux: requires x",
                    0, 0, "meyeraux", "", "numkit:meyeraux:nargin");
    outs[0] = meyeraux(args[0], ctx.engine->resource());
}

void shanwavf_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    if (args.size() < 5)
        throw Error("shanwavf: requires (LB, UB, N, fb, fc)",
                    0, 0, "shanwavf", "", "numkit:shanwavf:nargin");
    const double lb = args[0].toScalar();
    const double ub = args[1].toScalar();
    const size_t N  = static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "N", "shanwavf"), 3, "N", "shanwavf"));
    const double fb = args[3].toScalar();
    const double fc = args[4].toScalar();
    auto [psi, x] = shanwavf(lb, ub, N, fb, fc, ctx.engine->resource());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

void cmorwavf_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    if (args.size() < 3)
        throw Error("cmorwavf: requires (LB, UB, N[, fb, fc])",
                    0, 0, "cmorwavf", "", "numkit:cmorwavf:nargin");
    const double lb = args[0].toScalar();
    const double ub = args[1].toScalar();
    const size_t N  = static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "N", "cmorwavf"), 3, "N", "cmorwavf"));
    // MATLAB R2025b defaults when only 3 args supplied: fb = fc = 1.
    const double fb = (args.size() >= 4) ? args[3].toScalar() : 1.0;
    const double fc = (args.size() >= 5) ? args[4].toScalar() : 1.0;
    auto [psi, x] = cmorwavf(lb, ub, N, fb, fc, ctx.engine->resource());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

void fbspwavf_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    if (args.size() < 6)
        throw Error("fbspwavf: requires (LB, UB, N, m, fb, fc)",
                    0, 0, "fbspwavf", "", "numkit:fbspwavf:nargin");
    const double lb = args[0].toScalar();
    const double ub = args[1].toScalar();
    const size_t N  = static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "N", "fbspwavf"), 3, "N", "fbspwavf"));
    const int    m  = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[3], 4, "m", "fbspwavf"), 4, "m", "fbspwavf");
    const double fb = args[4].toScalar();
    const double fc = args[5].toScalar();
    auto [psi, x] = fbspwavf(lb, ub, N, m, fb, fc, ctx.engine->resource());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

void centfrq_reg(Span<const Value> args, size_t nargout,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "centfrq");
    if (!args[0].isChar() && !args[0].isString())
        throw Error("centfrq: WNAME must be a string or char vector",
                    0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
    const std::string wname = args[0].toString();
    int iter = 8;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        iter = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "ITER", "centfrq"),
            2, "ITER", "centfrq");
    }
    bool withRecfreq = false;
    if (args.size() >= 3) {
        if (!args[2].isChar() && !args[2].isString())
            throw Error("centfrq: third argument must be 'plot'",
                        0, 0, "centfrq", "", "numkit:centfrq:invalidPlotArg");
        withRecfreq = true;
    }
    auto res = centfrqGrid(wname, iter, withRecfreq, ctx.engine->resource());
    outs[0] = Value::scalar(res.freq, ctx.engine->resource());
    if (nargout > 1) outs[1] = std::move(res.xval);
    if (nargout > 2) outs[2] = std::move(res.recfreq);
}

void scal2frq_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "scal2frq");
    if (!args[1].isChar() && !args[1].isString())
        throw Error("scal2frq: WNAME must be a string or char vector",
                    0, 0, "scal2frq", "", "numkit:scal2frq:invalidWavelet");
    const std::string wname = args[1].toString();
    double delta = 1.0;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        delta = argcheck::expectedReal(args[2], 3, "DELTA", "scal2frq");
    }
    outs[0] = scal2frq(args[0], wname, delta, ctx.engine->resource());
}

void cwt_reg(Span<const Value> args, size_t nargout,
             Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 13, "cwt");
    std::string wname = "morse";
    double fs = 1.0;
    int voicesPerOctave = 10;
    double timeBandwidth = 60.0;
    double gamma = 3.0;
    bool extendSignal = true;
    Value freqLimits;

    auto lowerStr = [](std::string s) {
        for (char &c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };

    size_t idx = 1;
    // Parse optional positional wname and/or fs before Name-Value pairs.
    if (idx < args.size() && (args[idx].isChar() || args[idx].isString())) {
        const std::string s = lowerStr(args[idx].toString());
        if (s == "morse" || s == "amor" || s == "bump") {
            wname = s;
            ++idx;
        }
    }
    if (idx < args.size() && args[idx].isNumeric() && args[idx].numel() == 1) {
        fs = argcheck::expectedReal(args[idx], static_cast<int>(idx + 1), "FS", "cwt");
        ++idx;
    }

    while (idx < args.size()) {
        if (!args[idx].isChar() && !args[idx].isString())
            throw Error("cwt: expected option name at argument " + std::to_string(idx + 1),
                        0, 0, "cwt", "", "numkit:cwt:invalidOption");
        if (idx + 1 >= args.size())
            throw Error("cwt: missing value for option '" + args[idx].toString() + "'",
                        0, 0, "cwt", "", "numkit:cwt:missingOptionValue");
        const std::string key = lowerStr(args[idx].toString());
        const Value &val = args[idx + 1];
        if (key == "voicesperoctave") {
            voicesPerOctave = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(val, static_cast<int>(idx + 2), "VoicesPerOctave", "cwt"),
                static_cast<int>(idx + 2), "VoicesPerOctave", "cwt");
        } else if (key == "timebandwidth") {
            timeBandwidth = argcheck::expectedReal(val, static_cast<int>(idx + 2), "TimeBandwidth", "cwt");
        } else if (key == "waveletparameters") {
            if (!val.isNumeric() || val.isComplex() || val.numel() != 2)
                throw Error("cwt: WaveletParameters must be a 2-element real vector [gamma, TimeBandwidth]",
                            0, 0, "cwt", "", "numkit:cwt:invalidWaveletParameters");
            gamma = val.elemAsDouble(0);
            timeBandwidth = val.elemAsDouble(1);
        } else if (key == "extendsignal") {
            if (val.isLogical() || (val.isNumeric() && val.numel() == 1)) {
                extendSignal = val.toBool();
            } else {
                throw Error("cwt: ExtendSignal must be a logical scalar",
                            0, 0, "cwt", "", "numkit:cwt:invalidExtendSignal");
            }
        } else if (key == "frequencylimits") {
            freqLimits = val;
        } else {
            throw Error("cwt: unrecognized option '" + args[idx].toString() + "'",
                        0, 0, "cwt", "", "numkit:cwt:unknownOption");
        }
        idx += 2;
    }

    auto res = cwt(args[0], wname, fs, voicesPerOctave, timeBandwidth,
                   gamma, extendSignal, freqLimits, ctx.engine->resource());
    outs[0] = std::move(res.cfs);
    if (nargout > 1) outs[1] = std::move(res.f);
    if (nargout > 2) outs[2] = std::move(res.coi);
}

} // namespace detail
} // namespace numkit::wavelet

