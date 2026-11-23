// toolboxes/wavelet/src/dwt/dwt2_reg.cpp
//
// Register half of the 2-D DWT pair: the CallContext builtins dwt2 / idwt2
// (argument parsing, optional `sx` output-size handling, multi-output
// packing) that delegate to the engine-free compute in dwt2.cpp. Keeping
// this separate lets the compute build against value+fs+ops alone.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/wavelet/dwt/dwt2.hpp>

#include <numkit/sys/plugin.hpp>
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

void dwt2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "dwt2");
    auto *mr = ctx.engine->allocator();
    auto r = dwt2(args[0], argString(args[1]), mr);
    if (outs.size() >= 1) outs[0] = keel::move(r.cA);
    if (outs.size() >= 2) outs[1] = keel::move(r.cH);
    if (outs.size() >= 3) outs[2] = keel::move(r.cV);
    if (outs.size() >= 4) outs[3] = keel::move(r.cD);
}

void idwt2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 5, 6, "idwt2");
    keel::i64 outRows = -1, outCols = -1;
    if (args.size() >= 6 && !args[5].isEmpty()) {
        // sx form, probed R2025b: a length-2 vector [rows cols] (single
        // fine); a scalar sx is NOT accepted; a char sx is ignored (the
        // reference reconstructs at the natural size); complex rejects.
        const Value &sx = args[5];
        if (sx.isChar() || sx.isString()) {
            // ignored — natural size
        } else if (sx.type() == ValueType::COMPLEX) {
            throw Error("idwt2: input 6 (SIZ) must be real.",
                        0, 0, "idwt2", "", "numkit:idwt2:mustBeReal");
        } else if (sx.numel() == 1) {
            throw Error("idwt2: input 6 (SIZ) must be a two-element "
                        "[rows cols] size vector.",
                        0, 0, "idwt2", "", "numkit:idwt2:size");
        } else if (sx.numel() >= 2) {
            outRows = keel::as<keel::i64>(sx.elemAsDouble(0));
            outCols = keel::as<keel::i64>(sx.elemAsDouble(1));
        }
    }
    outs[0] = idwt2(args[0], args[1], args[2], args[3],
                    argString(args[4]), outRows, outCols,
                    ctx.engine->allocator());
}

} // namespace detail
} // namespace numkit::wavelet
