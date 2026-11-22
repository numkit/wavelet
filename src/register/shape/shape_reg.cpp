// toolboxes/wavelet/src/shape/shape_reg.cpp
//
// Register half of the continuous-wavelet shape primitives: the CallContext
// builtins mexihat / morlet / meyeraux / shanwavf / cmorwavf / fbspwavf
// (argument parsing + multi-output psi/x packing) that delegate to the
// engine-free compute in shape.cpp. shape_grid_reg is the shared
// register-side dispatcher. library.cpp forward-declares + registers these.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/wavelet/shape/shape.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <tuple>
#include <utility>

namespace numkit::wavelet {
namespace detail {

static void shape_grid_reg(const char *fn,
                           std::tuple<Value, Value> (*impl)(keel::f64, keel::f64, keel::usize, keel::Allocator *),
                           keel::ArrayRef<Value> args, keel::usize nargout,
                           keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, fn);
    const keel::f64 lb = args[0].toScalar();
    const keel::f64 ub = args[1].toScalar();
    const keel::f64 Nd = args[2].toScalar();
    if (!(Nd >= 0.0))
        throw Error(keel::format("{}: N must be non-negative", fn),
                    0, 0, fn, "", "numkit:wav:N");
    const keel::usize N = keel::as<keel::usize>(Nd);
    auto [psi, x] = impl(lb, ub, N, ctx.engine->allocator());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

void mexihat_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    shape_grid_reg("mexihat", &mexihat, args, nargout, outs, ctx);
}

void morlet_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    shape_grid_reg("morlet", &morlet, args, nargout, outs, ctx);
}

void meyeraux_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "meyeraux");
    outs[0] = meyeraux(args[0], ctx.engine->allocator());
}

void shanwavf_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 5, 5, "shanwavf");
    const keel::f64 lb = args[0].toScalar();
    const keel::f64 ub = args[1].toScalar();
    const keel::usize N  = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "N", "shanwavf"), 3, "N", "shanwavf"));
    const keel::f64 fb = args[3].toScalar();
    const keel::f64 fc = args[4].toScalar();
    auto [psi, x] = shanwavf(lb, ub, N, fb, fc, ctx.engine->allocator());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

void cmorwavf_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "cmorwavf");
    const keel::f64 lb = args[0].toScalar();
    const keel::f64 ub = args[1].toScalar();
    const keel::usize N  = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "N", "cmorwavf"), 3, "N", "cmorwavf"));
    // MATLAB R2025b defaults when only 3 args supplied: fb = fc = 1.
    const keel::f64 fb = (args.size() >= 4) ? args[3].toScalar() : 1.0;
    const keel::f64 fc = (args.size() >= 5) ? args[4].toScalar() : 1.0;
    auto [psi, x] = cmorwavf(lb, ub, N, fb, fc, ctx.engine->allocator());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

void fbspwavf_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 6, 6, "fbspwavf");
    const keel::f64 lb = args[0].toScalar();
    const keel::f64 ub = args[1].toScalar();
    const keel::usize N  = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "N", "fbspwavf"), 3, "N", "fbspwavf"));
    const keel::i32    m  = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[3], 4, "m", "fbspwavf"), 4, "m", "fbspwavf");
    const keel::f64 fb = args[4].toScalar();
    const keel::f64 fc = args[5].toScalar();
    auto [psi, x] = fbspwavf(lb, ub, N, m, fb, fc, ctx.engine->allocator());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

void centfrq_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "centfrq");
    if (!args[0].isChar() && !args[0].isString())
        throw Error("centfrq: WNAME must be a string or char vector",
                    0, 0, "centfrq", "", "numkit:centfrq:invalidWavelet");
    const keel::String wname = args[0].toString();
    keel::i32 iter = 8;
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
    auto res = centfrqGrid(wname, iter, withRecfreq, ctx.engine->allocator());
    outs[0] = Value::scalar(res.freq, ctx.engine->allocator());
    if (nargout > 1) outs[1] = std::move(res.xval);
    if (nargout > 2) outs[2] = std::move(res.recfreq);
}

void scal2frq_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "scal2frq");
    if (!args[1].isChar() && !args[1].isString())
        throw Error("scal2frq: WNAME must be a string or char vector",
                    0, 0, "scal2frq", "", "numkit:scal2frq:invalidWavelet");
    const keel::String wname = args[1].toString();
    keel::f64 delta = 1.0;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        delta = argcheck::expectedReal(args[2], 3, "DELTA", "scal2frq");
    }
    outs[0] = scal2frq(args[0], wname, delta, ctx.engine->allocator());
}

void cwt_reg(keel::ArrayRef<Value> args, keel::usize nargout,
             keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 13, "cwt");
    keel::String wname("morse");
    keel::f64 fs = 1.0;
    keel::i32 voicesPerOctave = 10;
    keel::f64 timeBandwidth = 60.0;
    keel::f64 gamma = 3.0;
    bool extendSignal = true;
    Value freqLimits;

    auto lowerStr = [](keel::StringRef sv) {
        keel::String s(sv);
        s.to_ascii_lower_in_place();
        return s;
    };

    keel::usize idx = 1;
    // Parse optional positional wname and/or fs before Name-Value pairs.
    if (idx < args.size() && (args[idx].isChar() || args[idx].isString())) {
        const keel::String raw = args[idx].toString();
        keel::String s = lowerStr(raw);
        if (s == "morse" || s == "amor" || s == "bump") {
            wname = std::move(s);
            ++idx;
        }
    }
    if (idx < args.size() && args[idx].isNumeric() && args[idx].numel() == 1) {
        fs = argcheck::expectedReal(args[idx], keel::as<keel::i32>(idx + 1), "FS", "cwt");
        ++idx;
    }

    while (idx < args.size()) {
        if (!args[idx].isChar() && !args[idx].isString())
            throw Error(keel::format("cwt: expected option name at argument {}", idx + 1),
                        0, 0, "cwt", "", "numkit:cwt:invalidOption");
        const keel::String optRaw = args[idx].toString();
        if (idx + 1 >= args.size())
            throw Error(keel::format("cwt: missing value for option '{}'", optRaw),
                        0, 0, "cwt", "", "numkit:cwt:missingOptionValue");
        const keel::String key = lowerStr(optRaw);
        const Value &val = args[idx + 1];
        if (key == "voicesperoctave") {
            voicesPerOctave = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(val, keel::as<keel::i32>(idx + 2), "VoicesPerOctave", "cwt"),
                keel::as<keel::i32>(idx + 2), "VoicesPerOctave", "cwt");
        } else if (key == "timebandwidth") {
            timeBandwidth = argcheck::expectedReal(val, keel::as<keel::i32>(idx + 2), "TimeBandwidth", "cwt");
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
            throw Error(keel::format("cwt: unrecognized option '{}'", optRaw),
                        0, 0, "cwt", "", "numkit:cwt:unknownOption");
        }
        idx += 2;
    }

    auto res = cwt(args[0], wname, fs, voicesPerOctave, timeBandwidth,
                   gamma, extendSignal, freqLimits, ctx.engine->allocator());
    outs[0] = std::move(res.cfs);
    if (nargout > 1) outs[1] = std::move(res.f);
    if (nargout > 2) outs[2] = std::move(res.coi);
}

} // namespace detail
} // namespace numkit::wavelet

