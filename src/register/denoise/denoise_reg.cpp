// toolboxes/wavelet/src/denoise/denoise_reg.cpp
//
// Register half of the thresholding / denoising primitives: the CallContext
// builtins wthresh / wnoisest / wdenoise that delegate to the engine-free
// compute in denoise.cpp. library.cpp forward-declares + registers these.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/wavelet/denoise/denoise.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

namespace numkit::wavelet {
namespace detail {

static keel::String argString(const Value &v) {
    if (!v.isChar() && !v.isString())
        throw Error("wavelet: expected string argument",
                    0, 0, "", "", "numkit:wavelet:type");
    return v.toString();
}

void wthresh_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "wthresh");
    outs[0] = wthresh(args[0], argString(args[1]), args[2],
                    ctx.engine->allocator());
}

void wnoisest_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "wnoisest");
    outs[0] = wnoisest(args[0], args[1], args[2], ctx.engine->allocator());
}

void wdenoise_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "wdenoise");
    keel::i32 level = -1;
    if (args.size() >= 2 && !args[1].isEmpty())
        level = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "level", "wdenoise"), 2, "level", "wdenoise");
    keel::String wname;
    if (args.size() >= 3 && !args[2].isEmpty())
        wname = argString(args[2]);
    outs[0] = wdenoise(args[0], level, wname, ctx.engine->allocator());
}

void wentropy_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "wentropy");
    const keel::f64 param = (args.size() >= 3) ? args[2].toScalar() : 0.0;
    outs[0] = wentropy(args[0], argString(args[1]), param, ctx.engine->allocator());
}

void ddencmp_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "ddencmp");
    auto r = ddencmp(argString(args[0]), argString(args[1]), args[2],
                     ctx.engine->allocator());
    if (outs.size() >= 1) outs[0] = std::move(r.thr);
    if (outs.size() >= 2) outs[1] = std::move(r.sorh);
    if (outs.size() >= 3) outs[2] = std::move(r.keepapp);
}

} // namespace detail
} // namespace numkit::wavelet
