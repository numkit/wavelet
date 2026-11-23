// toolboxes/wavelet/src/shape/gauss_reg.cpp
//
// Register half of the Gaussian wavelets: the CallContext builtins
// gauswavf / cgauwavf that delegate to the engine-free compute in
// gauss.cpp. parseGaussOrder lives here because it is a register-side
// parser (accepts an int order or a 'gausN'/'cgauN' wname string).
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/wavelet/shape/shape.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, ctx.engine->resource()
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <cerrno>
#include <cstdlib>
#include <utility>

namespace numkit::wavelet {
namespace detail {

// Parse the optional `p` argument: either an integer (1..8) or a string
// of the form `'gausN'` (real form) / `'cgauN'` (complex form). The
// `prefix` is the expected wname stem.
static keel::i32 parseGaussOrder(const Value &arg, const char *prefix, const char *fn)
{
    if (arg.isChar() || arg.isString()) {
        keel::String s = arg.toString();
        // Lowercase the prefix portion for comparison.
        s.to_ascii_lower_in_place();
        const keel::StringRef pref(prefix);
        if (s.size() <= pref.size() || !s.starts_with(pref))
            throw Error(keel::format("{}: bad wname '{}' (expected {}N)", fn, s, prefix),
                         0, 0, fn, "", "numkit:wname");
        const char *numStart = s.c_str() + pref.size();
        char *numEnd = nullptr;
        errno = 0;
        const keel::i64 v = std::strtol(numStart, &numEnd, 10);
        if (numEnd == numStart || *numEnd != '\0' || errno == ERANGE ||
            v < keel::Limits<keel::i32>::kMin || v > keel::Limits<keel::i32>::kMax) {
            throw Error(keel::format("{}: bad wname '{}' (cannot parse order N)", fn, s),
                         0, 0, fn, "", "numkit:wname");
        }
        return keel::as<keel::i32>(v);
    }
    return argcheck::expectedInteger(
        argcheck::expectedReal(arg, 1, "order", "gauswavf"), 1, "order", "gauswavf");
}

void gauswavf_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "gauswavf");
    const keel::f64 lb = args[0].toScalar();
    const keel::f64 ub = args[1].toScalar();
    const keel::usize N  = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "N", "gauswavf"), 3, "N", "gauswavf"));
    keel::i32 p = 1;
    if (args.size() >= 4) p = parseGaussOrder(args[3], "gaus", "gauswavf");
    auto [psi, x] = gauswavf(lb, ub, N, p, ctx.engine->allocator());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

void cgauwavf_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "cgauwavf");
    const keel::f64 lb = args[0].toScalar();
    const keel::f64 ub = args[1].toScalar();
    const keel::usize N  = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "N", "cgauwavf"), 3, "N", "cgauwavf"));
    keel::i32 p = 1;
    if (args.size() >= 4) p = parseGaussOrder(args[3], "cgau", "cgauwavf");
    auto [psi, x] = cgauwavf(lb, ub, N, p, ctx.engine->allocator());
    outs[0] = std::move(psi);
    if (nargout > 1) outs[1] = std::move(x);
}

} // namespace detail
} // namespace numkit::wavelet
