// toolboxes/wavelet/src/dwt/multilevel_reg.cpp
//
// Register half of the multi-level DWT family: the CallContext builtins
// wavedec / waverec / appcoef / detcoef (argument parsing, custom-filter
// vs wname dispatch, 'mode' N-V handling, 'cells' / multi-level detcoef
// shapes) that delegate to the engine-free compute in multilevel.cpp.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/wavelet/dwt/multilevel.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace numkit::wavelet {
namespace detail {

static std::string argString(const Value &v) {
    if (!v.isChar() && !v.isString())
        throw Error("wavelet: expected string argument",
                    0, 0, "", "", "numkit:wavelet:type");
    return v.toString();
}

void wavedec_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                 CallContext &ctx)
{
    if (args.size() < 3)
        throw Error("wavedec: requires (x, n, wname)",
                    0, 0, "wavedec", "", "numkit:wavedec:nargin");
    auto *mr = ctx.engine->resource();
    auto [C, L] = wavedec(args[0], argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "level", "wavedec"), 2, "level", "wavedec"),
                          argString(args[2]), mr);
    if (outs.size() >= 1) outs[0] = std::move(C);
    if (outs.size() >= 2) outs[1] = std::move(L);
}

void waverec_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                 CallContext &ctx)
{
    if (args.size() < 3)
        throw Error("waverec: requires (C, L, wname)",
                    0, 0, "waverec", "", "numkit:waverec:nargin");
    outs[0] = waverec(args[0], args[1], argString(args[2]),
                      ctx.engine->resource());
}

void appcoef_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                 CallContext &ctx)
{
    if (args.size() < 3)
        throw Error("appcoef: requires (C, L, wname[, level]) or "
                    "(C, L, Lo_R, Hi_R[, level])",
                    0, 0, "appcoef", "", "numkit:appcoef:nargin");
    auto *mr = ctx.engine->resource();

    // Parse trailing 'mode' / Mode= N-V: only 'sym' supported.
    auto lower = [](std::string s) {
        for (auto &c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    auto check_mode_at = [&](size_t start) {
        for (size_t i = start; i + 1 < args.size(); i += 2) {
            if (!(args[i].isChar() || args[i].isString())) break;
            const std::string key = lower(args[i].toString());
            if (key == "mode") {
                const std::string m = lower(args[i + 1].toString());
                if (m != "sym" && m != "symh")
                    throw Error("appcoef: only 'mode'='sym' is implemented "
                                "(got '" + m + "')",
                                0, 0, "appcoef", "", "numkit:appcoef:mode_nyi");
            }
        }
    };

    int level = -1;
    auto readVec = [](const Value &v) {
        const size_t n = v.numel();
        std::vector<double> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
        return out;
    };

    if (args[2].isChar() || args[2].isString()) {
        // (C, L, wname[, level][, 'mode', extmode])
        size_t i = 3;
        if (i < args.size() && !args[i].isEmpty()
            && args[i].numel() == 1
            && !(args[i].isChar() || args[i].isString())) {
            level = (int)args[i].toScalar();
            ++i;
        }
        check_mode_at(i);
        outs[0] = appcoef(args[0], args[1], args[2].toString(), level, mr);
    } else {
        // (C, L, Lo_R, Hi_R[, level][, 'mode', extmode])
        if (args.size() < 4 || (args[3].isChar() || args[3].isString()))
            throw Error("appcoef: custom-filter form requires "
                        "(C, L, Lo_R, Hi_R)",
                        0, 0, "appcoef", "", "numkit:appcoef:nargin");
        size_t i = 4;
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

void detcoef_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                 CallContext &ctx)
{
    if (args.size() < 2)
        throw Error("detcoef: requires (C, L[, level[, 'cells']])",
                    0, 0, "detcoef", "", "numkit:detcoef:nargin");
    auto *mr = ctx.engine->resource();
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
            std::string s = args[3].toString();
            for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s == "cells";
        }();

    if (hasCellsFlag) {
        // 'cells' form: build a cell array, one detail per requested level.
        const size_t k = levArg.numel();
        Value cellOut = Value::cell(1, k);
        for (size_t i = 0; i < k; ++i) {
            const int lev = static_cast<int>(levArg.elemAsDouble(i));
            cellOut.cellAt(i) = detcoef(args[0], args[1], lev, mr);
        }
        outs[0] = std::move(cellOut);
        return;
    }

    const long long maxLev = static_cast<long long>(args[1].numel()) - 2;

    // detcoef(C, L, 'cells'): cell array of ALL levels 1..nMax.
    if (levArg.isChar() || levArg.isString()) {
        std::string s = levArg.toString();
        for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s != "cells")
            throw Error("detcoef: unknown option '" + levArg.toString() + "'",
                        0, 0, "detcoef", "", "numkit:detcoef:opt");
        Value cellOut = Value::cell(1, static_cast<size_t>(std::max<long long>(0, maxLev)));
        for (int lev = 1; lev <= maxLev; ++lev)
            cellOut.cellAt(static_cast<size_t>(lev - 1)) =
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
    const size_t k = levArg.numel();
    if (nargout >= 2) {
        for (size_t i = 0; i < k && i < outs.size(); ++i)
            outs[i] = detcoef(args[0], args[1],
                              static_cast<int>(levArg.elemAsDouble(i)), mr);
    } else {
        Value cellOut = Value::cell(1, k);
        for (size_t i = 0; i < k; ++i)
            cellOut.cellAt(i) =
                detcoef(args[0], args[1],
                        static_cast<int>(levArg.elemAsDouble(i)), mr);
        outs[0] = std::move(cellOut);
    }
}

void wenergy_reg(Span<const Value> args, size_t, Span<Value> outs, CallContext &ctx)
{
    if (args.size() < 2)
        throw Error("wenergy: requires (C, L)",
                    0, 0, "wenergy", "", "numkit:wenergy:nargin");
    auto r = wenergy(args[0], args[1], ctx.engine->resource());
    if (outs.size() >= 1) outs[0] = std::move(r.Ea);
    if (outs.size() >= 2) outs[1] = std::move(r.Ed);
}

void upcoef_reg(Span<const Value> args, size_t, Span<Value> outs, CallContext &ctx)
{
    if (args.size() < 3)
        throw Error("upcoef: requires (O, X, wname [, N [, L]])",
                    0, 0, "upcoef", "", "numkit:upcoef:nargin");
    const std::string type = args[0].toString();
    const std::string wname = args[2].toString();
    // N = 0 is the documented "derive the level count from X" form
    // (upcoef with zero levels is the identity — see UpcoefTest).
    const int n = (args.size() >= 4)
        ? argcheck::expectedInteger(
        argcheck::expectedReal(args[3], 4, "n", "upcoef"), 4, "n", "upcoef") : 1;
    if (args.size() >= 4)
        argcheck::expectedNonNegative(double(n), 4, "n", "upcoef");
    const long long len =
        (args.size() >= 5) ? static_cast<long long>(args[4].toScalar()) : -1;
    outs[0] = upcoef(type, args[1], wname, n, len, ctx.engine->resource());
}

void wpdec_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "wpdec");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "N", "wpdec"), 2, "N", "wpdec");
    const std::string wname = argString(args[2]);
    const std::string entName = (args.size() >= 4) ? argString(args[3]) : "shannon";
    const double entPar = (args.size() >= 5 && !args[4].isEmpty())
                              ? argcheck::expectedReal(args[4], 5, "P", "wpdec")
                              : 0.0;
    outs[0] = wpdec(args[0], n, wname, entName, entPar, ctx.engine->resource());
}

void wpcoef_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "wpcoef");
    const Value node = (args.size() >= 2) ? args[1] : Value::scalar(0.0);
    outs[0] = wpcoef(args[0], node, ctx.engine->resource());
}

void wprec_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "wprec");
    outs[0] = wprec(args[0], ctx.engine->resource());
}

void wpsplt_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "wpsplt");
    auto res = wpsplt(args[0], args[1], ctx.engine->resource());
    outs[0] = std::move(res.tree);
    if (nargout > 1) outs[1] = std::move(res.cA);
    if (nargout > 2) outs[2] = std::move(res.cD);
}

void besttree_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "besttree");
    auto res = besttree(args[0], ctx.engine->resource());
    outs[0] = std::move(res.tree);
    if (nargout > 1) outs[1] = std::move(res.entropy);
}

} // namespace detail
} // namespace numkit::wavelet
