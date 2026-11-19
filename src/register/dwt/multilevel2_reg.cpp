// toolboxes/wavelet/src/dwt/multilevel2_reg.cpp
//
// Register the 2-D multilevel DWT family: the CallContext builtins
// wavedec2 / waverec2 / appcoef2 / detcoef2 (argument parsing, the
// detcoef2 'all' 3-output shape, default-level handling) that delegate to
// the engine-free compute in multilevel2.cpp. library.cpp forward-declares
// + registers these by name.

#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/wavelet/dwt/multilevel2.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <utility>

namespace numkit::wavelet {
namespace detail {

static keel::String argString(const Value &v) {
    if (!v.isChar() && !v.isString())
        throw Error("wavelet: expected string argument",
                    0, 0, "", "", "numkit:wavelet:type");
    return v.toString();
}

void wavedec2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "wavedec2");
    auto *mr = ctx.engine->allocator();
    auto r = wavedec2(args[0], argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "level", "wavedec2"), 2, "level", "wavedec2"),
                      argString(args[2]), mr);
    if (outs.size() >= 1) outs[0] = std::move(r.c);
    if (outs.size() >= 2) outs[1] = std::move(r.s);
}

void waverec2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "waverec2");
    outs[0] = waverec2(args[0], args[1], argString(args[2]),
                       ctx.engine->allocator());
}

void appcoef2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "appcoef2");
    auto *mr = ctx.engine->allocator();
    int level = -1;   // default = coarsest
    if (args.size() >= 4 && !args[3].isEmpty()
        && !(args[3].isChar() || args[3].isString()))
        level = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[3], 4, "level", "appcoef2"), 4, "level", "appcoef2");
    outs[0] = appcoef2(args[0], args[1], argString(args[2]), level, mr);
}

void detcoef2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "detcoef2");
    auto *mr = ctx.engine->allocator();
    keel::String type = argString(args[0]);
    keel::String lt = type.clone();
    for (auto &c : lt) c = keel::to_ascii_lower(c);
    const int level = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[3], 4, "level", "detcoef2"), 4, "level", "detcoef2");

    if (lt == "all") {
        // [H, V, D] = detcoef2('all', C, S, level).
        if (outs.size() >= 1) outs[0] = detcoef2("h", args[1], args[2], level, mr);
        if (outs.size() >= 2) outs[1] = detcoef2("v", args[1], args[2], level, mr);
        if (outs.size() >= 3) outs[2] = detcoef2("d", args[1], args[2], level, mr);
        return;
    }
    outs[0] = detcoef2(type, args[1], args[2], level, mr);
}

} // namespace detail
} // namespace numkit::wavelet
