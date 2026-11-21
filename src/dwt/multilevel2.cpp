// toolboxes/wavelet/src/dwt/multilevel2.cpp
// 2-D multilevel DWT — wavedec2 / waverec2 / appcoef2 / detcoef2, delegated to keel::wavelet.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/wavelet.h>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/wavelet/dwt/multilevel2.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>

#include <cstring>

namespace numkit::wavelet {

namespace {

keel::wavelet::FilterBank lookup_keel_fb(keel::StringRef wname, keel::Allocator *mr)
{
    auto nfb = wavelet_filters(wname);
    keel::wavelet::FilterBank kfb{
        keel::Array<keel::f64>(mr),
        keel::Array<keel::f64>(mr),
        keel::Array<keel::f64>(mr),
        keel::Array<keel::f64>(mr),
    };
    kfb.lo_d.assign(nfb.Lo_D);
    kfb.hi_d.assign(nfb.Hi_D);
    kfb.lo_r.assign(nfb.Lo_R);
    kfb.hi_r.assign(nfb.Hi_R);
    return kfb;
}

void readS(const Value &S, keel::Array<keel::usize> &s_rows, keel::Array<keel::usize> &s_cols)
{
    const keel::usize R = S.dims().rows();
    s_rows.resize(R);
    s_cols.resize(R);
    for (keel::usize i = 0; i < R; ++i) {
        s_rows[i] = keel::as<keel::usize>(S.elemAsDouble(i));
        s_cols[i] = keel::as<keel::usize>(S.elemAsDouble(R + i));
    }
}

void wavedec2_argcheck(keel::i32 n)
{
    if (n < 1)
        throw Error("wavedec2: level must be >= 1",
                    0, 0, "wavedec2", "", "numkit:wavedec2:level");
}

keel::i32 detcoef2_argcheck(keel::StringRef type, const Value &C, const Value &S, keel::i32 level)
{
    internal::requireWaveClass(C, "detcoef2", "C", 2,
                               internal::WvClasses::AnyNumeric,
                               /*allowComplex=*/true);
    internal::requireWaveClass(S, "detcoef2", "S", 3,
                               internal::WvClasses::AnyNumeric);
    const keel::i32 N = keel::as<keel::i32>(S.dims().rows()) - 2;
    if (N < 1)
        throw Error("detcoef2: S must have at least 3 rows (1 level)",
                    0, 0, "detcoef2", "", "numkit:detcoef2:size");
    if (level < 1 || level > N)
        throw Error("detcoef2: level out of range [1, N]",
                    0, 0, "detcoef2", "", "numkit:detcoef2:level");

    if (type == "h" || type == "H") return 0;
    if (type == "v" || type == "V") return 1;
    if (type == "d" || type == "D") return 2;
    throw Error(keel::format("detcoef2: type must be 'h', 'v' or 'd' (got '{}')", type),
                0, 0, "detcoef2", "", "numkit:detcoef2:type");
}

} // anonymous

Wavedec2Result wavedec2(const Value &X, keel::i32 n, keel::StringRef wname,
                        keel::Allocator *mr)
{
    wavedec2_argcheck(n);
    auto fb = lookup_keel_fb(wname, mr);
    const keel::usize M = X.dims().rows();
    const keel::usize N = X.dims().cols();
    keel::Array<keel::f64> xv(M * N, mr);
    for (keel::usize i = 0; i < M * N; ++i) xv[i] = X.elemAsDouble(i);

    auto r = keel::wavelet::wavedec2(xv, M, N, keel::as<keel::i32>(n), fb, mr);
    const keel::usize R = r.s_rows.size();
    Value S = Value::matrix(R, 2, ValueType::DOUBLE, mr);
    keel::f64 *sd = S.doubleDataMut();
    for (keel::usize i = 0; i < R; ++i) {
        sd[i] = keel::as<keel::f64>(r.s_rows[i]);
        sd[R + i] = keel::as<keel::f64>(r.s_cols[i]);
    }
    Value c = Value::matrix(1, r.c.size(), ValueType::DOUBLE, mr);
    if (!r.c.empty()) std::memcpy(c.doubleDataMut(), r.c.data(), r.c.size() * sizeof(keel::f64));
    return { std::move(c), std::move(S) };
}

static Value appcoef2_impl(const Value &C, const Value &S,
                           keel::StringRef wname, keel::i32 level,
                           keel::Allocator *mr)
{
    const keel::i32 N = keel::as<keel::i32>(S.dims().rows()) - 2;
    if (N < 1)
        throw Error("appcoef2: S must have at least 3 rows (1 level)",
                    0, 0, "appcoef2", "", "numkit:appcoef2:size");
    if (level < 0) level = N;
    if (level < 0 || level > N)
        throw Error("appcoef2: level out of range [0, N]",
                    0, 0, "appcoef2", "", "numkit:appcoef2:level");

    auto fb = lookup_keel_fb(wname, mr);
    keel::Array<keel::f64> cv(C.numel(), mr);
    for (keel::usize i = 0; i < C.numel(); ++i) cv[i] = C.elemAsDouble(i);
    keel::Array<keel::usize> sr(mr), sc(mr);
    readS(S, sr, sc);

    auto r = keel::wavelet::appcoef2(cv, sr, sc, fb, keel::as<keel::i32>(level), mr);
    Value m = Value::matrix(r.rows, r.cols, ValueType::DOUBLE, mr);
    if (!r.out.empty()) std::memcpy(m.doubleDataMut(), r.out.data(), r.out.size() * sizeof(keel::f64));
    return m;
}

Value waverec2(const Value &C, const Value &S, keel::StringRef wname,
               keel::Allocator *mr)
{
    return appcoef2_impl(C, S, wname, /*level=*/0, mr);
}

Value appcoef2(const Value &C, const Value &S, keel::StringRef wname,
               keel::i32 level, keel::Allocator *mr)
{
    return appcoef2_impl(C, S, wname, level, mr);
}

Value detcoef2(keel::StringRef type, const Value &C, const Value &S,
               keel::i32 level, keel::Allocator *mr)
{
    const keel::i32 which = detcoef2_argcheck(type, C, S, level);
    const keel::i32 N = keel::as<keel::i32>(S.dims().rows()) - 2;
    const keel::usize R = S.dims().rows();

    auto sRow = [&](keel::usize i) -> std::pair<keel::usize, keel::usize> {
        return {keel::as<keel::usize>(S.elemAsDouble(i)),
                keel::as<keel::usize>(S.elemAsDouble(R + i))};
    };

    auto [aR, aC] = sRow(0);
    keel::usize off = aR * aC;
    for (keel::i32 lev = N; lev > level; --lev) {
        auto [dR, dC] = sRow(keel::as<keel::usize>(N - lev + 1));
        off += 3 * dR * dC;
    }
    auto [dR, dC] = sRow(keel::as<keel::usize>(N - level + 1));
    const keel::usize dn = dR * dC;
    off += keel::as<keel::usize>(which) * dn;
    if (off + dn > C.numel()) {
        throw Error(keel::format("detcoef2: C is too short for the sizes in S "
                                 "(needed {} elements, got {}).",
                                 off + dn, C.numel()),
                    0, 0, "detcoef2", "", "numkit:detcoef2:size");
    }

    if (C.type() == ValueType::DOUBLE) {
        keel::Array<keel::usize> sr(mr), sc(mr);
        readS(S, sr, sc);
        auto r = keel::wavelet::detcoef2(
            static_cast<keel::wavelet::Detail2D>(which),
            keel::ArrayRef<keel::f64>(C.doubleData(), C.numel()),
            sr, sc, keel::as<keel::i32>(level), mr);
        Value out = Value::matrix(r.rows, r.cols, ValueType::DOUBLE, mr);
        if (!r.out.empty()) std::memcpy(out.doubleDataMut(), r.out.data(), r.out.size() * sizeof(keel::f64));
        return out;
    }

    const bool cplx = C.type() == ValueType::COMPLEX;
    Value out = cplx ? Value::complexMatrix(dR, dC, mr)
                     : Value::matrix(dR, dC, C.type(), mr);
    for (keel::usize i = 0; i < dn; ++i) {
        const keel::usize src = off + i;
        if (cplx) out.complexDataMut()[i] = C.complexElem(src);
        else internal::storeElem(out, i, C.elemAsDouble(src));
    }
    return out;
}

} // namespace numkit::wavelet
