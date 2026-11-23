// toolboxes/wavelet/src/dwt/dwt.cpp
// Single-level discrete wavelet transform and its inverse — Value adapter
// delegating numerical transforms to keel::wavelet::{dwt, idwt}.

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/dwt/dwt.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::wavelet {

namespace {

FilterBank dwt_argcheck(keel::StringRef wname) {
    return wavelet_filters(wname);
}

void dwt_argcheck(keel::ArrayRef<keel::f64> Lo_D,
                  keel::ArrayRef<keel::f64> Hi_D) {
    const keel::usize Lf = Lo_D.size();
    if (Lf < 2 || Hi_D.size() != Lf)
        throw Error("dwt: filter length < 2 or Lo_D/Hi_D length mismatch",
                    0, 0, "dwt", "", "numkit:dwt:filt");
}

FilterBank idwt_argcheck(keel::StringRef wname) {
    return wavelet_filters(wname);
}

void idwt_argcheck(keel::ArrayRef<keel::f64> Lo_R,
                   keel::ArrayRef<keel::f64> Hi_R) {
    const keel::usize Lf = Lo_R.size();
    if (Lf < 2 || Hi_R.size() != Lf)
        throw Error("idwt: filter length < 2 or Lo_R/Hi_R length mismatch",
                    0, 0, "idwt", "", "numkit:idwt:filt");
}

} // anonymous

static keel::Pair<Value, Value>
dwt_with_filters(const Value &x,
                 keel::ArrayRef<keel::f64> Lo_D,
                 keel::ArrayRef<keel::f64> Hi_D,
                 keel::Allocator *mr)
{
    dwt_argcheck(Lo_D, Hi_D);
    const keel::usize N = x.numel();
    if (N == 0)
        return {Value::matrix(1, 0, ValueType::DOUBLE, mr),
                Value::matrix(1, 0, ValueType::DOUBLE, mr)};
    keel::Array<keel::f64> xv(N, mr);
    for (keel::usize i = 0; i < N; ++i) xv[i] = x.elemAsDouble(i);

    auto res = keel::wavelet::dwt(keel::ArrayRef<keel::f64>(xv.data(), N), Lo_D, Hi_D, mr);
    auto pack = [&](keel::ArrayRef<keel::f64> v) {
        Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
        if (!v.empty()) keel::copy_n(r.doubleDataMut(), v.data(), v.size());
        return r;
    };
    return {pack(keel::ArrayRef<keel::f64>(res.ca.data(), res.ca.size())),
            pack(keel::ArrayRef<keel::f64>(res.cd.data(), res.cd.size()))};
}

keel::Pair<Value, Value>
dwt(const Value &x, keel::StringRef wname,
    keel::Allocator *mr)
{
    auto fb = dwt_argcheck(wname);
    return dwt_with_filters(x, fb.Lo_D, fb.Hi_D, mr);
}

keel::Pair<Value, Value>
dwt_with_filters_pub(const Value &x,
                     keel::ArrayRef<keel::f64> Lo_D,
                     keel::ArrayRef<keel::f64> Hi_D,
                     keel::Allocator *mr)
{
    dwt_argcheck(Lo_D, Hi_D);
    return dwt_with_filters(x, Lo_D, Hi_D, mr);
}

Value idwt_with_filters(const Value &cA, const Value &cD,
                        keel::ArrayRef<keel::f64> Lo_R,
                        keel::ArrayRef<keel::f64> Hi_R,
                        keel::i64 len,
                        keel::Allocator *mr)
{
    idwt_argcheck(Lo_R, Hi_R);
    const keel::usize la = cA.numel();
    const keel::usize ld = cD.numel();
    keel::Array<keel::f64> av(la, mr);
    keel::Array<keel::f64> dv(ld, mr);
    for (keel::usize i = 0; i < la; ++i) av[i] = cA.elemAsDouble(i);
    for (keel::usize i = 0; i < ld; ++i) dv[i] = cD.elemAsDouble(i);

    auto rec = keel::wavelet::idwt(keel::ArrayRef<keel::f64>(av.data(), la),
                                   keel::ArrayRef<keel::f64>(dv.data(), ld),
                                   Lo_R, Hi_R, keel::as<keel::i64>(len), mr);
    const keel::usize outN = rec.size();
    const bool wantSingle = cA.type() == ValueType::SINGLE;
    Value r = Value::matrix(1, outN,
                            wantSingle ? ValueType::SINGLE
                                       : ValueType::DOUBLE,
                            mr);
    if (wantSingle) {
        keel::f32 *rf = r.singleDataMut();
        for (keel::usize i = 0; i < outN; ++i) rf[i] = keel::as<keel::f32>(rec[i]);
    } else if (outN > 0) {
        keel::copy_n(r.doubleDataMut(), rec.data(), outN);
    }
    return r;
}

Value idwt(const Value &cA, const Value &cD,
           keel::StringRef wname,
           keel::i64 len,
           keel::Allocator *mr)
{
    auto fb = idwt_argcheck(wname);
    return idwt_with_filters(cA, cD, fb.Lo_R, fb.Hi_R, len, mr);
}

Value idwt_with_filters_pub(const Value &cA, const Value &cD,
                            keel::ArrayRef<keel::f64> Lo_R,
                            keel::ArrayRef<keel::f64> Hi_R,
                            keel::i64 len,
                            keel::Allocator *mr)
{
    idwt_argcheck(Lo_R, Hi_R);
    return idwt_with_filters(cA, cD, Lo_R, Hi_R, len, mr);
}

} // namespace numkit::wavelet
