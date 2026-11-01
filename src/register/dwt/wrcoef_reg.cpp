// toolboxes/wavelet/src/dwt/wrcoef_reg.cpp
//
// Register half of wrcoef: the CallContext builtin (type/level parsing,
// wname vs custom-filter dispatch) that delegates to the engine-free compute in
// wrcoef.cpp. The compute wrcoef() is declared in multilevel.hpp, so this
// TU includes that header. library.cpp forward-declares + registers this.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/wavelet/dwt/multilevel.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <cctype>
#include <cmath>
#include <string>
#include <vector>

namespace numkit::wavelet {
namespace detail {

void wrcoef_reg(Span<const Value> args, size_t /*nargout*/,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 6, "wrcoef");

    if (!args[0].isChar() && !args[0].isString())
        throw Error("wrcoef: type must be a character vector ('a' or 'd')",
                    0, 0, "wrcoef", "", "numkit:wrcoef:type");
    const std::string type = args[0].toString();

    auto parse_level = [](const Value &v) -> int {
        if (v.isEmpty()) return -1;
        const double nd = v.toScalar();
        if (nd < 0.0 || nd != std::floor(nd))
            throw Error("wrcoef: level n must be a non-negative integer",
                        0, 0, "wrcoef", "", "numkit:wrcoef:level");
        return static_cast<int>(nd);
    };

    if (args[3].isChar() || args[3].isString()) {
        argcheck::checkNargin(args.size(), 4, 5, "wrcoef");
        const std::string wname = args[3].toString();
        const int n = (args.size() >= 5) ? parse_level(args[4]) : -1;
        outs[0] = wrcoef(type, args[1], args[2], wname, n,
                         ctx.engine->resource());
    } else {
        argcheck::checkNargin(args.size(), 5, 6, "wrcoef");
        if (!args[3].isNumeric() || args[3].isComplex() ||
            !args[4].isNumeric() || args[4].isComplex()) {
            throw Error("wrcoef: Lo_R and Hi_R must be real numeric filter vectors",
                        0, 0, "wrcoef", "", "numkit:wrcoef:filt");
        }
        auto read_vec = [](const Value &v) {
            std::vector<double> out(v.numel());
            for (size_t i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
            return out;
        };
        const int n = (args.size() >= 6) ? parse_level(args[5]) : -1;
        outs[0] = wrcoef(type, args[1], args[2],
                         read_vec(args[3]), read_vec(args[4]), n,
                         ctx.engine->resource());
    }
}

} // namespace detail
} // namespace numkit::wavelet
