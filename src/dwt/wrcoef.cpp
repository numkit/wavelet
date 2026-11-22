// toolboxes/wavelet/src/dwt/wrcoef.cpp
// 1-D single-band reconstruction — wrcoef, delegated to keel::wavelet::wrcoef.

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <keel/wavelet.h>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/wavelet/dwt/multilevel.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>


namespace numkit::wavelet {

namespace {

keel::String lower_type(keel::StringRef sv)
{
    keel::String s(sv);
    s.to_ascii_lower_in_place();
    return s;
}

keel::String wrcoef_argcheck(keel::StringRef type, const Value &c,
                             const Value &l, keel::i32 n)
{
    keel::String tl = lower_type(type);
    if (tl != "a" && tl != "d")
        throw Error("wrcoef: type must be 'a' or 'd'",
                    0, 0, "wrcoef", "", "numkit:wrcoef:type");
    if (n < -1)
        throw Error("wrcoef: level n must be a non-negative integer",
                    0, 0, "wrcoef", "", "numkit:wrcoef:level");

    const keel::usize Lcount = l.numel();
    if (Lcount < 3)
        throw Error("wrcoef: l must have at least 3 entries",
                    0, 0, "wrcoef", "", "numkit:wrcoef:l");
    const keel::i32 N = keel::as<keel::i32>(Lcount) - 2;
    const keel::i32 effN = (n == -1) ? N : n;

    if (tl == "a") {
        if (effN < 0 || effN > N)
            throw Error(keel::format("wrcoef: for type='a', level must satisfy 0 <= n <= {}", N),
                        0, 0, "wrcoef", "", "numkit:wrcoef:level");
    } else {
        if (effN < 1 || effN > N)
            throw Error(keel::format("wrcoef: for type='d', level must satisfy 1 <= n <= {}", N),
                        0, 0, "wrcoef", "", "numkit:wrcoef:level");
    }

    keel::usize expectedC = keel::as<keel::usize>(l.elemAsDouble(0));
    for (keel::i32 k = 0; k < N; ++k)
        expectedC += keel::as<keel::usize>(l.elemAsDouble(1 + k));
    if (c.numel() != expectedC)
        throw Error("wrcoef: c length does not match l bookkeeping",
                    0, 0, "wrcoef", "", "numkit:wrcoef:cl");
    return tl;
}

keel::String wrcoef_argcheck(keel::StringRef type, const Value &c,
                             const Value &l,
                             keel::ArrayRef<keel::f64> Lo_R,
                             keel::ArrayRef<keel::f64> Hi_R,
                             keel::i32 n)
{
    keel::String tl = wrcoef_argcheck(type, c, l, n);
    if (Lo_R.size() < 2 || Hi_R.size() != Lo_R.size())
        throw Error("wrcoef: filter length < 2 or Lo_R/Hi_R length mismatch",
                    0, 0, "wrcoef", "", "numkit:wrcoef:filt");
    return tl;
}

Value wrcoef_reconstruct_impl(keel::StringRef tl, const Value &c,
                              const Value &l,
                              keel::ArrayRef<keel::f64> Lo_R,
                              keel::ArrayRef<keel::f64> Hi_R,
                              keel::i32 n,
                              keel::Allocator *mr)
{
    keel::Array<keel::f64> cv(c.numel(), mr);
    for (keel::usize i = 0; i < c.numel(); ++i) cv[i] = c.elemAsDouble(i);
    keel::Array<keel::usize> lv(l.numel(), mr);
    for (keel::usize i = 0; i < l.numel(); ++i) lv[i] = keel::as<keel::usize>(l.elemAsDouble(i));

    auto rec = keel::wavelet::wrcoef(tl == "d", cv, lv, Lo_R, Hi_R, keel::as<keel::i32>(n), mr);
    Value r = Value::matrix(1, rec.size(), ValueType::DOUBLE, mr);
    if (!rec.empty()) keel::copy_n(r.doubleDataMut(), rec.data(), rec.size());
    return r;
}

} // anonymous

Value wrcoef(keel::StringRef type, const Value &c, const Value &l,
             keel::StringRef wname, keel::i32 n,
             keel::Allocator *mr)
{
    const keel::String tl = wrcoef_argcheck(type, c, l, n);
    auto fb = wavelet_filters(wname);
    return wrcoef_reconstruct_impl(tl, c, l, fb.Lo_R, fb.Hi_R, n, mr);
}

Value wrcoef(keel::StringRef type, const Value &c, const Value &l,
             keel::ArrayRef<keel::f64> Lo_R,
             keel::ArrayRef<keel::f64> Hi_R,
             keel::i32 n,
             keel::Allocator *mr)
{
    const keel::String tl = wrcoef_argcheck(type, c, l, Lo_R, Hi_R, n);
    return wrcoef_reconstruct_impl(tl, c, l, Lo_R, Hi_R, n, mr);
}

} // namespace numkit::wavelet
