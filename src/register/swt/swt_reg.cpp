// toolboxes/wavelet/src/swt/swt_reg.cpp
//
// Register half of the stationary / maximal-overlap transforms: the
// CallContext builtins swt / iswt / modwt / imodwt (argument parsing,
// modwt default wname/level inference) that delegate to the engine-free
// compute in swt.cpp. library.cpp forward-declares + registers these.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/wavelet/swt/swt.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <cmath>
#include <string>
#include <utility>

namespace numkit::wavelet {
namespace detail {

static std::string argString(const Value &v) {
    if (!v.isChar() && !v.isString())
        throw Error("wavelet: expected string argument",
                    0, 0, "", "", "numkit:wavelet:type");
    return v.toString();
}

void swt_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
             CallContext &ctx)
{
    if (args.size() < 3)
        throw Error("swt: requires (x, n, wname)",
                    0, 0, "swt", "", "numkit:swt:nargin");
    const int lev = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "level", "swt"), 2, "level", "swt");
    auto *mr = ctx.engine->resource();
    if (nargout >= 2) {
        // [SWA, SWD] = swt(x, n, wname) — per-level approximation and
        // detail matrices.
        auto [swa, swd] = swtAD(args[0], lev, argString(args[2]), mr);
        outs[0] = std::move(swa);
        if (outs.size() >= 2) outs[1] = std::move(swd);
        return;
    }
    outs[0] = swt(args[0], lev, argString(args[2]), mr);
}

void iswt_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
              CallContext &ctx)
{
    // Reference dispatch (iswt.m): 2 args = (SWC, wname);
    // 3 args = (SWA, SWD, wname) when the 3rd is text, else
    // (SWC, Lo_R, Hi_R); 4 args = (SWA, SWD, Lo_R, Hi_R). The
    // custom-filter forms are stubbed until the compute kernel grows a
    // filter-vector entry point.
    auto *mr = ctx.engine->resource();
    if (args.size() < 2)
        throw Error("iswt: requires (swc, wname) or (swa, swd, wname)",
                    0, 0, "iswt", "", "numkit:iswt:nargin");
    const bool a3text = args.size() >= 3 && (args[2].isChar()
                                             || args[2].isString());
    const bool a2text = args[1].isChar() || args[1].isString();
    if (args.size() == 2 && a2text) {
        outs[0] = iswt(args[0], argString(args[1]), mr);
        return;
    }
    if (args.size() == 3 && a3text && !a2text) {
        outs[0] = iswt(args[0], args[1], argString(args[2]), mr);
        return;
    }
    if (args.size() == 3 && !a3text && !a2text)
        throw Error("iswt: custom reconstruction filters are not "
                    "supported yet (pass a wavelet name).",
                    0, 0, "iswt", "", "numkit:iswt:optionUnsupported");
    if (args.size() == 4 && !a2text && !a3text)
        throw Error("iswt: custom reconstruction filters are not "
                    "supported yet (pass a wavelet name).",
                    0, 0, "iswt", "", "numkit:iswt:optionUnsupported");
    throw Error("iswt: expected (swc, wname) or (swa, swd, wname)",
                0, 0, "iswt", "", "numkit:iswt:nargin");
}

void modwt_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
               CallContext &ctx)
{
    // MATLAB R2025b signatures:
    //   w = modwt(x)
    //   w = modwt(x, wname)
    //   w = modwt(x, wname, lev)
    //   w = modwt(x, lev)            (numeric 2nd arg → uses default wname)
    // Default wname = 'sym4'; default lev = floor(log2(N)).
    if (args.empty())
        throw Error("modwt: requires (x[, wname[, lev]])",
                    0, 0, "modwt", "", "numkit:modwt:nargin");
    auto *mr = ctx.engine->resource();
    const Value &x = args[0];
    const size_t N = x.numel();
    const int default_lev = (N >= 2)
        ? static_cast<int>(std::floor(std::log2(static_cast<double>(N))))
        : 1;

    std::string wname = "sym4";
    int lev = default_lev;
    if (args.size() >= 2) {
        if (args[1].isChar() || args[1].isString()) {
            wname = args[1].toString();
            if (args.size() >= 3)
                lev = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "level", "modwt"), 3, "level", "modwt");
        } else {
            // Numeric 2nd arg: that's the level; a trailing char 3rd arg
            // is the wname (the reference's modwt(x, lev, wname) form).
            lev = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "level", "modwt"), 2, "level", "modwt");
            if (args.size() >= 3 && (args[2].isChar() || args[2].isString()))
                wname = args[2].toString();
        }
    }
    outs[0] = modwt(x, lev, wname, mr);
}

void imodwt_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                CallContext &ctx)
{
    if (args.size() < 2)
        throw Error("imodwt: requires (swc, wname)",
                    0, 0, "imodwt", "", "numkit:imodwt:nargin");
    outs[0] = imodwt(args[0], argString(args[1]), ctx.engine->resource());
}

} // namespace detail
} // namespace numkit::wavelet
