// toolboxes/wavelet/src/filter/families.cpp
//
// Family-named scaling filters (dbwavf / coifwavf / symwavf) and
// orthogonal filter quadruple (orthfilt).
//
// MATLAB convention (verified vs R2025b):
//   dbwavf('dbN')     = Lo_R / sqrt(2)        (length 2N, sum = 1)
//   coifwavf('coifK') = Lo_R / sqrt(2)        (length 6K)
//   symwavf('symN')   = Lo_R / sqrt(2)        (length 2N)
// where Lo_R is the synthesis lowpass returned by wfilters('xxx').
//
// orthfilt(W) takes a unit-normalised scaling filter W (sum(W) = 1)
// and emits the four filter banks in MATLAB output order
// [Lo_D, Hi_D, Lo_R, Hi_R]:
//
//   Lo_R[k]    = W[k] * sqrt(2)
//   Lo_D[k]    = Lo_R[N-1-k]                       (time reversal)
//   Hi_R[k]    = (-1)^k * Lo_R[N-1-k]              (QMF, alt-sign reversal)
//   Hi_D[k]    = Hi_R[N-1-k]                       (= reverse(Hi_R))
//
// Internally this lib stores Lo_D (MATLAB naming) under the variable
// `Lo_R` of the FilterBank struct (a long-standing legacy of the
// initial wfilters.cpp commit). Code below treats `fb.Lo_R` as
// "MATLAB Lo_D" and reverses it to obtain MATLAB Lo_R, matching the
// rest of MATLAB's documented output.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/filter/families.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>

// Compute-only TU: Value substrate + Error, no engine. The dbwavf /
// coifwavf / symwavf / orthfilt builtins (CallContext wrappers) live in
// filter/families_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>

namespace numkit::wavelet {

namespace {

const keel::f64 INV_SQRT2 = 0.70710678118654752440;

Value rowVec(keel::Allocator *mr, keel::ArrayRef<keel::f64> v)
{
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

// dbwavf / coifwavf / symwavf share the same body — emit Lo_R / sqrt(2).
// 2026-05-08: with the wfilters Lo_D/Lo_R label fix, fb.Lo_R is now the
// MATLAB Lo_R directly (was MATLAB Lo_D under nk's old swapped
// labels). No reversal needed anymore.
Value family_scaling(keel::Allocator *mr, keel::StringRef name)
{
    auto fb = wavelet_filters(name);   // throws on unsupported family
    keel::Array<keel::f64> v = std::move(fb.Lo_R);
    for (auto &x : v) x *= INV_SQRT2;
    return rowVec(mr, v);
}

void dbwavf_argcheck(keel::StringRef name)
{
    if (!name.starts_with("db") && name != "haar")
        throw Error(keel::format("dbwavf: name must be 'haar' or 'dbN' (got '{}')", name),
                    0, 0, "dbwavf", "", "numkit:dbwavf:name");
}

void coifwavf_argcheck(keel::StringRef name)
{
    if (!name.starts_with("coif"))
        throw Error(keel::format("coifwavf: name must be 'coifK' (got '{}')", name),
                    0, 0, "coifwavf", "", "numkit:coifwavf:name");
}

void symwavf_argcheck(keel::StringRef name)
{
    if (!name.starts_with("sym"))
        throw Error(keel::format("symwavf: name must be 'symN' (got '{}')", name),
                    0, 0, "symwavf", "", "numkit:symwavf:name");
}

void orthfilt_argcheck(const Value &W)
{
    if (W.numel() == 0)
        throw Error("orthfilt: scaling filter must be non-empty",
                    0, 0, "orthfilt", "", "numkit:orthfilt:empty");
}

} // anonymous

// ── Public C++ API (see filter/families.hpp) ──────────────────────────

Value dbwavf(keel::StringRef name, keel::Allocator *mr)
{
    dbwavf_argcheck(name);
    return family_scaling(mr, name);
}

Value coifwavf(keel::StringRef name, keel::Allocator *mr)
{
    coifwavf_argcheck(name);
    return family_scaling(mr, name);
}

Value symwavf(keel::StringRef name, keel::Allocator *mr)
{
    symwavf_argcheck(name);
    return family_scaling(mr, name);
}

OrthfiltResult orthfilt(const Value &W, keel::Allocator *mr)
{
    orthfilt_argcheck(W);
    const keel::usize N = W.numel();
    keel::Array<keel::f64> wv(N, mr);
    for (keel::usize k = 0; k < N; ++k) wv[k] = W.elemAsDouble(k);
    auto fb = keel::wavelet::orthfilt(keel::ArrayRef<keel::f64>(wv.data(), N), mr);
    return { rowVec(mr, fb.lo_d), rowVec(mr, fb.hi_d),
             rowVec(mr, fb.lo_r), rowVec(mr, fb.hi_r) };
}

} // namespace numkit::wavelet
