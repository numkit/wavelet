// toolboxes/wavelet/src/dwt/multilevel_reg.cpp
//
// Register half of the multi-level DWT family: the CallContext builtins
// wavedec / waverec / appcoef / detcoef (argument parsing, custom-filter
// vs wname dispatch, 'mode' N-V handling, 'cells' / multi-level detcoef
// shapes) that delegate to the engine-free compute in multilevel.cpp.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/wavelet/dwt/multilevel.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <utility>

namespace numkit::wavelet {
namespace detail {

static keel::String argString(const Value &v) {
    if (!v.isChar() && !v.isString())
        throw Error("wavelet: expected string argument",
                    0, 0, "", "", "numkit:wavelet:type");
    return v.toString();
}

void wavedec_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "wavedec");
    auto *mr = ctx.engine->allocator();
    auto [C, L] = wavedec(args[0], argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "level", "wavedec"), 2, "level", "wavedec"),
                          argString(args[2]), mr);
    if (outs.size() >= 1) outs[0] = std::move(C);
    if (outs.size() >= 2) outs[1] = std::move(L);
}

void waverec_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "waverec");
    outs[0] = waverec(args[0], args[1], argString(args[2]),
                      ctx.engine->allocator());
}

void appcoef_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 7, "appcoef");
    auto *mr = ctx.engine->allocator();

    // Parse trailing 'mode' / Mode= N-V: only 'sym' supported.
    auto lower = [](keel::StringRef sv) {
        keel::String s(sv);
        for (auto &c : s) c = keel::to_ascii_lower(c);
        return s;
    };
    auto check_mode_at = [&](keel::usize start) {
        for (keel::usize i = start; i + 1 < args.size(); i += 2) {
            if (!(args[i].isChar() || args[i].isString())) break;
            const keel::String keyRaw = args[i].toString();
            const keel::String key = lower(keyRaw);
            if (key == "mode") {
                const keel::String mRaw = args[i + 1].toString();
                const keel::String m = lower(mRaw);
                if (m != "sym" && m != "symh")
                    throw Error(keel::format("appcoef: only 'mode'='sym' is implemented "
                                             "(got '{}')", m),
                                0, 0, "appcoef", "", "numkit:appcoef:mode_nyi");
            }
        }
    };

    int level = -1;
    auto readVec = [](const Value &v) {
        const keel::usize n = v.numel();
        keel::Array<double> out(n);
        for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
        return out;
    };

    if (args[2].isChar() || args[2].isString()) {
        // (C, L, wname[, level][, 'mode', extmode])
        keel::usize i = 3;
        if (i < args.size() && !args[i].isEmpty()
            && args[i].numel() == 1
            && !(args[i].isChar() || args[i].isString())) {
            level = (int)args[i].toScalar();
            ++i;
        }
        check_mode_at(i);
        const keel::String wname = args[2].toString();
        outs[0] = appcoef(args[0], args[1], wname, level, mr);
    } else {
        // (C, L, Lo_R, Hi_R[, level][, 'mode', extmode])
        if (args.size() < 4 || (args[3].isChar() || args[3].isString()))
            throw Error("appcoef: custom-filter form requires "
                        "(C, L, Lo_R, Hi_R)",
                        0, 0, "appcoef", "", "numkit:appcoef:nargin");
        keel::usize i = 4;
        if (i < args.size() && !args[i].isEmpty()
            && args[i].numel() == 1
            && !(args[i].isChar() || args[i].isString())) {
            level = (int)args[i].toScalar();
            ++i;
        }
        check_mode_at(i);
        outs[0] = appcoef_with_filters_pub(args[0], args[1],
                                           readVec(args[2]), readVec(args[3]),
                                           level, mr);
    }
}

void detcoef_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "detcoef");
    auto *mr = ctx.engine->allocator();
    // Default level = max (deepest decomposition level) when not supplied.
    // MATLAB R2025b: detcoef(c, l) returns the level numel(L) - 2 detail.
    // Bug fix 2026-05-08: was throwing on the 2-arg form.
    if (args.size() == 2) {
        const long long maxLev =
            static_cast<long long>(args[1].numel()) - 2;
        if (maxLev < 1)
            throw Error("detcoef: L is too short to infer a default level",
                        0, 0, "detcoef", "", "numkit:detcoef:size");
        outs[0] = detcoef(args[0], args[1], static_cast<int>(maxLev), mr);
        return;
    }

    // Detect 'cells' form: levels arg is a vector AND 4th arg is 'cells'.
    const Value &levArg = args[2];
    const bool hasCellsFlag = (args.size() >= 4)
        && (args[3].isChar() || args[3].isString())
        && [&] {
            keel::String s = args[3].toString();
            for (auto &c : s) c = keel::to_ascii_lower(c);
            return s == "cells";
        }();

    if (hasCellsFlag) {
        // 'cells' form: build a cell array, one detail per requested level.
        const keel::usize k = levArg.numel();
        Value cellOut = Value::cell(1, k);
        for (keel::usize i = 0; i < k; ++i) {
            const int lev = static_cast<int>(levArg.elemAsDouble(i));
            cellOut.cellAt(i) = detcoef(args[0], args[1], lev, mr);
        }
        outs[0] = std::move(cellOut);
        return;
    }

    const long long maxLev = static_cast<long long>(args[1].numel()) - 2;

    // detcoef(C, L, 'cells'): cell array of ALL levels 1..nMax.
    if (levArg.isChar() || levArg.isString()) {
        const keel::String orig = levArg.toString();
        keel::String s = orig.clone();
        for (auto &c : s) c = keel::to_ascii_lower(c);
        if (s != "cells")
            throw Error(keel::format("detcoef: unknown option '{}'", orig),
                        0, 0, "detcoef", "", "numkit:detcoef:opt");
        Value cellOut = Value::cell(1, static_cast<keel::usize>(std::max<long long>(0, maxLev)));
        for (int lev = 1; lev <= maxLev; ++lev)
            cellOut.cellAt(static_cast<keel::usize>(lev - 1)) =
                detcoef(args[0], args[1], lev, mr);
        outs[0] = std::move(cellOut);
        return;
    }

    // Single-level scalar form (existing behavior).
    if (levArg.numel() == 1) {
        outs[0] = detcoef(args[0], args[1], argcheck::expectedPositiveInteger(
            argcheck::expectedReal(levArg, 3, "level", "detcoef"),
            3, "level", "detcoef"), mr);
        return;
    }

    // Vector of levels: detcoef(C, L, [n1 n2 ...]). With a single (or no)
    // output, return a cell array of the per-level details; with multiple
    // outputs, deal one detail per output ([d1, d2, ...] = detcoef(...)).
    const keel::usize k = levArg.numel();
    if (nargout >= 2) {
        for (keel::usize i = 0; i < k && i < outs.size(); ++i)
            outs[i] = detcoef(args[0], args[1],
                              static_cast<int>(levArg.elemAsDouble(i)), mr);
    } else {
        Value cellOut = Value::cell(1, k);
        for (keel::usize i = 0; i < k; ++i)
            cellOut.cellAt(i) =
                detcoef(args[0], args[1],
                        static_cast<int>(levArg.elemAsDouble(i)), mr);
        outs[0] = std::move(cellOut);
    }
}

void wenergy_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "wenergy");
    auto r = wenergy(args[0], args[1], ctx.engine->allocator());
    if (outs.size() >= 1) outs[0] = std::move(r.Ea);
    if (outs.size() >= 2) outs[1] = std::move(r.Ed);
}

void upcoef_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "upcoef");
    const keel::String type = args[0].toString();
    const keel::String wname = args[2].toString();
    // N = 0 is the documented "derive the level count from X" form
    // (upcoef with zero levels is the identity — see UpcoefTest).
    const int n = (args.size() >= 4)
        ? argcheck::expectedInteger(
        argcheck::expectedReal(args[3], 4, "n", "upcoef"), 4, "n", "upcoef") : 1;
    if (args.size() >= 4)
        argcheck::expectedNonNegative(double(n), 4, "n", "upcoef");
    const long long len =
        (args.size() >= 5) ? static_cast<long long>(args[4].toScalar()) : -1;
    outs[0] = upcoef(type, args[1], wname, n, len, ctx.engine->allocator());
}

void wpdec_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "wpdec");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "N", "wpdec"), 2, "N", "wpdec");
    const keel::String wname = argString(args[2]);
    const keel::String entName = (args.size() >= 4) ? argString(args[3]) : keel::String("shannon");
    const double entPar = (args.size() >= 5 && !args[4].isEmpty())
                              ? argcheck::expectedReal(args[4], 5, "P", "wpdec")
                              : 0.0;
    outs[0] = wpdec(args[0], n, wname, entName, entPar, ctx.engine->allocator());
}

void wpcoef_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "wpcoef");
    const Value node = (args.size() >= 2) ? args[1] : Value::scalar(0.0);
    outs[0] = wpcoef(args[0], node, ctx.engine->allocator());
}

void wprec_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "wprec");
    outs[0] = wprec(args[0], ctx.engine->allocator());
}

void wpsplt_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "wpsplt");
    auto res = wpsplt(args[0], args[1], ctx.engine->allocator());
    outs[0] = std::move(res.tree);
    if (nargout > 1) outs[1] = std::move(res.cA);
    if (nargout > 2) outs[2] = std::move(res.cD);
}

void besttree_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "besttree");
    auto res = besttree(args[0], ctx.engine->allocator());
    outs[0] = std::move(res.tree);
    if (nargout > 1) outs[1] = std::move(res.entropy);
}

} // namespace detail
} // namespace numkit::wavelet
