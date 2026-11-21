// toolboxes/wavelet/src/dwt/dwt_reg.cpp
//
// Register half of the single-level DWT pair: the CallContext builtins
// dwt / idwt (argument parsing, custom-filter vs wname dispatch, optional
// `len` and 'mode' N-V handling) that delegate to the engine-free compute
// in dwt.cpp. This is the only DWT TU that needs the engine; keeping it
// separate lets the compute build against value+fs+ops alone.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/wavelet/dwt/dwt.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <tuple>

namespace numkit::wavelet {
namespace detail {

// Read a Value (numeric vector) into a flat double buffer.
static keel::Array<keel::f64> readVec(const Value &v, keel::Allocator *mr) {
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, mr);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

// Walk trailing N-V pairs starting at args[start]; returns true if a
// 'mode' arg specifies anything other than 'sym'. Throws on unsupported
// boundary modes (only 'sym' implemented for now).
static bool parse_mode_nv(keel::ArrayRef<Value> args, keel::usize start,
                          const char *fn) {
    auto lower = [](keel::StringRef sv) {
        keel::String s(sv);
        for (auto &c : s) c = keel::to_ascii_lower(c);
        return s;
    };
    for (keel::usize i = start; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString())) break;
        const keel::String rawKey = args[i].toString();
        const keel::String key = lower(rawKey);
        if (key == "mode") {
            const keel::String rawMode = args[i + 1].toString();
            const keel::String m = lower(rawMode);
            if (m != "sym" && m != "symh") {
                throw Error(keel::format("{}: only 'mode'='sym' is implemented (got '{}')", fn, m),
                            0, 0, fn, "", "numkit:wavelet:mode_nyi");
            }
        }
    }
    return false;
}

void dwt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
             CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "dwt");
    auto *mr = ctx.engine->allocator();
    Value cA, cD;
    if (args[1].isChar() || args[1].isString()) {
        // dwt(x, wname[, 'mode', extmode])
        parse_mode_nv(args, 2, "dwt");
        const keel::String wname = args[1].toString();
        std::tie(cA, cD) = dwt(args[0], wname, mr);
    } else {
        // dwt(x, Lo_D, Hi_D[, 'mode', extmode])
        if (args.size() < 3 || (args[2].isChar() || args[2].isString()))
            throw Error("dwt: custom-filter form requires (x, Lo_D, Hi_D)",
                        0, 0, "dwt", "", "numkit:dwt:nargin");
        parse_mode_nv(args, 3, "dwt");
        std::tie(cA, cD) = dwt_with_filters_pub(args[0],
                                                readVec(args[1], mr), readVec(args[2], mr),
                                                mr);
    }
    if (outs.size() >= 1) outs[0] = cA;
    if (outs.size() >= 2) outs[1] = cD;
}

void idwt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 7, "idwt");
    auto *mr = ctx.engine->allocator();
    long long len = -1;

    if (args[2].isChar() || args[2].isString()) {
        // idwt(cA, cD, wname[, len][, 'mode', extmode])
        const keel::String wname = args[2].toString();
        keel::usize i = 3;
        // Optional positional `len` (numeric scalar) BEFORE any N-V.
        if (i < args.size() && args[i].numel() == 1
            && !(args[i].isChar() || args[i].isString())) {
            len = (long long)args[i].toScalar();
            ++i;
        }
        parse_mode_nv(args, i, "idwt");
        outs[0] = idwt(args[0], args[1], wname, len, mr);
    } else {
        // idwt(cA, cD, Lo_R, Hi_R[, len][, 'mode', extmode])
        if (args.size() < 4 || (args[3].isChar() || args[3].isString()))
            throw Error("idwt: custom-filter form requires "
                        "(cA, cD, Lo_R, Hi_R)",
                        0, 0, "idwt", "", "numkit:idwt:nargin");
        keel::usize i = 4;
        if (i < args.size() && args[i].numel() == 1
            && !(args[i].isChar() || args[i].isString())) {
            len = (long long)args[i].toScalar();
            ++i;
        }
        parse_mode_nv(args, i, "idwt");
        outs[0] = idwt_with_filters_pub(args[0], args[1],
                                        readVec(args[2], mr), readVec(args[3], mr),
                                        len, mr);
    }
}

} // namespace detail
} // namespace numkit::wavelet
