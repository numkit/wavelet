// toolboxes/wavelet/src/dwt/dwt2.cpp
// 2-D separable wavelet transform — Value adapter delegating to keel::wavelet::{dwt2, idwt2}.

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/wavelet.h>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/wavelet/dwt/dwt2.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>


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

void idwt2_argcheck(const Value &cA, const Value &cH,
                    const Value &cV, const Value &cD)
{
    const Value *coefs[4] = {&cA, &cH, &cV, &cD};
    const char *names[4] = {"cA", "cH", "cV", "cD"};
    for (keel::i32 k = 0; k < 4; ++k)
        internal::requireWaveClass(*coefs[k], "idwt2", names[k], k + 1,
                                   internal::WvClasses::AnyNumeric,
                                   /*allowComplex=*/true);
    for (keel::i32 k = 1; k < 4; ++k) {
        if (coefs[k]->dims().rows() != cA.dims().rows()
            || coefs[k]->dims().cols() != cA.dims().cols())
            throw Error(keel::format("idwt2: cA, cH, cV and cD must have the same size (got {} and {}).",
                                     internal::shapeOf(cA), internal::shapeOf(*coefs[k])),
                        0, 0, "idwt2", "", "numkit:idwt2:size");
    }
}

Value packMatrix(const keel::Array<keel::f64> &buf, keel::usize rows, keel::usize cols,
                 keel::Allocator *mr)
{
    Value m = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!buf.empty()) keel::copy_n(m.doubleDataMut(), buf.data(), buf.size());
    return m;
}

} // anonymous

Dwt2Result dwt2(const Value &X, keel::StringRef wname,
                keel::Allocator *mr)
{
    auto fb = lookup_keel_fb(wname, mr);
    const keel::usize M = X.dims().rows();
    const keel::usize N = X.dims().cols();
    if (M == 0 || N == 0) {
        Value empty = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return {empty, empty, empty, empty};
    }

    keel::Array<keel::f64> xv(M * N, mr);
    for (keel::usize i = 0; i < M * N; ++i) xv[i] = X.elemAsDouble(i);

    auto r = keel::wavelet::dwt2(xv, M, N, fb, mr);
    return {
        packMatrix(r.ca, r.out_rows, r.out_cols, mr),
        packMatrix(r.ch, r.out_rows, r.out_cols, mr),
        packMatrix(r.cv, r.out_rows, r.out_cols, mr),
        packMatrix(r.cd, r.out_rows, r.out_cols, mr),
    };
}

Value idwt2(const Value &cA, const Value &cH,
            const Value &cV, const Value &cD,
            keel::StringRef wname,
            keel::i64 outRows, keel::i64 outCols,
            keel::Allocator *mr)
{
    idwt2_argcheck(cA, cH, cV, cD);
    const Value *coefs[4] = {&cA, &cH, &cV, &cD};

    const bool allEmpty = cA.numel() == 0 && cH.numel() == 0
        && cV.numel() == 0 && cD.numel() == 0;
    if (allEmpty)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    auto fb = lookup_keel_fb(wname, mr);
    const keel::usize rows = cA.dims().rows();
    const keel::usize cols = cA.dims().cols();
    const keel::usize lf = fb.lo_r.size();

    auto natural_len = [lf](keel::usize la) -> keel::usize {
        const keel::usize sx = 2 * la + lf - 2;
        const keel::i64 want = keel::as<keel::i64>(2 * la) - keel::as<keel::i64>(lf) + 2;
        if (want < 0 || keel::as<keel::usize>(want) >= sx) return sx;
        return keel::as<keel::usize>(want);
    };
    const keel::usize natR = natural_len(rows);
    const keel::usize natC = natural_len(cols);
    if ((outRows >= 0 && keel::as<keel::usize>(outRows) > natR)
        || (outCols >= 0 && keel::as<keel::usize>(outCols) > natC)) {
        throw Error(keel::format("idwt2: input 6 (SIZ) exceeds the natural "
                                 "reconstruction size {}x{}.", natR, natC),
                    0, 0, "idwt2", "", "numkit:idwt2:size");
    }

    bool anySngl = false, anyCplx = false;
    for (const Value *c : coefs) {
        anySngl |= c->type() == ValueType::SINGLE;
        anyCplx |= c->type() == ValueType::COMPLEX;
    }

    const keel::usize nBand = rows * cols;
    auto runPart = [&](bool imag) {
        keel::Array<keel::f64> b[4];
        for (keel::i32 k = 0; k < 4; ++k) {
            b[k] = keel::Array<keel::f64>(nBand, 0.0, mr);
            if (!imag) {
                for (keel::usize i = 0; i < nBand; ++i) b[k][i] = coefs[k]->elemAsDouble(i);
            } else if (coefs[k]->type() == ValueType::COMPLEX) {
                for (keel::usize i = 0; i < nBand; ++i) b[k][i] = coefs[k]->complexElem(i).imag();
            }
        }
        return keel::wavelet::idwt2(b[0], b[1], b[2], b[3], rows, cols, fb,
                                    keel::as<keel::i64>(outRows),
                                    keel::as<keel::i64>(outCols), mr);
    };

    auto resR = runPart(false);
    if (!anyCplx) {
        if (anySngl) {
            Value s = Value::matrix(resR.rows, resR.cols, ValueType::SINGLE, mr);
            keel::f32 *sd = s.singleDataMut();
            for (keel::usize i = 0; i < resR.out.size(); ++i)
                sd[i] = keel::as<keel::f32>(resR.out[i]);
            return s;
        }
        return packMatrix(resR.out, resR.rows, resR.cols, mr);
    }

    auto resI = runPart(true);
    Value y = Value::complexMatrix(resR.rows, resR.cols, mr);
    keel::c64 *yd = y.complexDataMut();
    for (keel::usize i = 0; i < resR.out.size(); ++i)
        yd[i] = keel::c64(resR.out[i], resI.out[i]);
    return y;
}

} // namespace numkit::wavelet
