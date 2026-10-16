// toolboxes/wavelet/src/dwt/dwt2.cpp
//
// 2-D separable wavelet transform. Built on the cycle-26 1-D dwt /
// idwt: apply a row pass, then a column pass (or the reverse for the
// inverse). MATLAB output order is [cA, cH, cV, cD] with the labels:
//
//   cA (approx)            = Lo-rows then Lo-cols   (LL)
//   cH (horizontal detail) = Lo-rows then Hi-cols   (LH)
//   cV (vertical detail)   = Hi-rows then Lo-cols   (HL)
//   cD (diagonal detail)   = Hi-rows then Hi-cols   (HH)
//
// Internally we slice rows/columns as length-K Value vectors, hand
// them to dwt / idwt, and stitch the results back into MxN matrices.

#include <numkit/wavelet/dwt/dwt2.hpp>
#include <numkit/wavelet/dwt/dwt.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The dwt2 / idwt2
// builtins (CallContext wrappers) live in dwt/dwt2_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace numkit::wavelet {

namespace {

// Apply 1-D dwt to every row of X. Returns (Lo, Hi), each M × outCols.
std::pair<Value, Value>
dwt_rows(const Value &X, const std::string &wname,
         std::pmr::memory_resource *mr)
{
    const size_t M = X.dims().rows();
    const size_t N = X.dims().cols();
    if (M == 0 || N == 0)
        return {Value::matrix(M, 0, ValueType::DOUBLE, mr),
                Value::matrix(M, 0, ValueType::DOUBLE, mr)};

    // Determine the per-row output length by running one row first.
    auto row = Value::matrix(1, N, ValueType::DOUBLE, mr);
    double *rd = row.doubleDataMut();
    for (size_t c = 0; c < N; ++c) rd[c] = X.elemAsDouble(c * M + 0);
    auto [cA0, cD0] = dwt(row, wname, mr);
    const size_t outN = cA0.numel();

    Value loM = Value::matrix(M, outN, ValueType::DOUBLE, mr);
    Value hiM = Value::matrix(M, outN, ValueType::DOUBLE, mr);
    double *ld = loM.doubleDataMut();
    double *hd = hiM.doubleDataMut();
    // First row already done.
    {
        const double *aL = cA0.doubleData();
        const double *aH = cD0.doubleData();
        for (size_t c = 0; c < outN; ++c) {
            ld[c * M + 0] = aL[c];
            hd[c * M + 0] = aH[c];
        }
    }
    for (size_t r = 1; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) rd[c] = X.elemAsDouble(c * M + r);
        auto [cA, cD] = dwt(row, wname, mr);
        const double *aL = cA.doubleData();
        const double *aH = cD.doubleData();
        for (size_t c = 0; c < outN; ++c) {
            ld[c * M + r] = aL[c];
            hd[c * M + r] = aH[c];
        }
    }
    return {std::move(loM), std::move(hiM)};
}

// Apply 1-D dwt to every column of X. Returns (Lo, Hi), each outRows × N.
std::pair<Value, Value>
dwt_cols(const Value &X, const std::string &wname,
         std::pmr::memory_resource *mr)
{
    const size_t M = X.dims().rows();
    const size_t N = X.dims().cols();
    if (M == 0 || N == 0)
        return {Value::matrix(0, N, ValueType::DOUBLE, mr),
                Value::matrix(0, N, ValueType::DOUBLE, mr)};

    auto col = Value::matrix(M, 1, ValueType::DOUBLE, mr);
    double *cd = col.doubleDataMut();
    for (size_t r = 0; r < M; ++r) cd[r] = X.elemAsDouble(0 * M + r);
    auto [cA0, cD0] = dwt(col, wname, mr);
    const size_t outM = cA0.numel();

    Value loM = Value::matrix(outM, N, ValueType::DOUBLE, mr);
    Value hiM = Value::matrix(outM, N, ValueType::DOUBLE, mr);
    double *ld = loM.doubleDataMut();
    double *hd = hiM.doubleDataMut();
    {
        const double *aL = cA0.doubleData();
        const double *aH = cD0.doubleData();
        for (size_t r = 0; r < outM; ++r) {
            ld[0 * outM + r] = aL[r];
            hd[0 * outM + r] = aH[r];
        }
    }
    for (size_t c = 1; c < N; ++c) {
        for (size_t r = 0; r < M; ++r) cd[r] = X.elemAsDouble(c * M + r);
        auto [cA, cD] = dwt(col, wname, mr);
        const double *aL = cA.doubleData();
        const double *aH = cD.doubleData();
        for (size_t r = 0; r < outM; ++r) {
            ld[c * outM + r] = aL[r];
            hd[c * outM + r] = aH[r];
        }
    }
    return {std::move(loM), std::move(hiM)};
}

// Inverse: combine a (lo, hi) pair into a single matrix by running
// 1-D idwt along each column, with target length outRows.
Value idwt_cols(const Value &Lo, const Value &Hi,
                const std::string &wname,
                size_t outRows,
                std::pmr::memory_resource *mr)
{
    const size_t M = Lo.dims().rows();
    const size_t N = Lo.dims().cols();
    Value out = Value::matrix(outRows, N, ValueType::DOUBLE, mr);
    if (M == 0 || N == 0 || outRows == 0) return out;
    double *od = out.doubleDataMut();

    auto colA = Value::matrix(M, 1, ValueType::DOUBLE, mr);
    auto colD = Value::matrix(M, 1, ValueType::DOUBLE, mr);
    double *ad = colA.doubleDataMut();
    double *dd = colD.doubleDataMut();
    for (size_t c = 0; c < N; ++c) {
        for (size_t r = 0; r < M; ++r) {
            ad[r] = Lo.elemAsDouble(c * M + r);
            dd[r] = Hi.elemAsDouble(c * M + r);
        }
        Value y = idwt(colA, colD, wname,
                       static_cast<long long>(outRows), mr);
        const double *yd = y.doubleData();
        for (size_t r = 0; r < outRows; ++r)
            od[c * outRows + r] = yd[r];
    }
    return out;
}

// Inverse along rows, target length outCols.
Value idwt_rows(const Value &Lo, const Value &Hi,
                const std::string &wname,
                size_t outCols,
                std::pmr::memory_resource *mr)
{
    const size_t M = Lo.dims().rows();
    const size_t N = Lo.dims().cols();
    Value out = Value::matrix(M, outCols, ValueType::DOUBLE, mr);
    if (M == 0 || N == 0 || outCols == 0) return out;
    double *od = out.doubleDataMut();

    auto rowA = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    auto rowD = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    double *ad = rowA.doubleDataMut();
    double *dd = rowD.doubleDataMut();
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            ad[c] = Lo.elemAsDouble(c * M + r);
            dd[c] = Hi.elemAsDouble(c * M + r);
        }
        Value y = idwt(rowA, rowD, wname,
                       static_cast<long long>(outCols), mr);
        const double *yd = y.doubleData();
        for (size_t c = 0; c < outCols; ++c)
            od[c * M + r] = yd[c];
    }
    return out;
}

} // anonymous

Dwt2Result dwt2(const Value &X, const std::string &wname,
                std::pmr::memory_resource *mr)
{
    // Pass 1: rows (each row → Lo + Hi halves, each M × outN).
    auto [LoR, HiR] = dwt_rows(X, wname, mr);

    // Pass 2: columns of LoR  → cA / cH; columns of HiR → cV / cD.
    auto [Lo_of_LoR, Hi_of_LoR] = dwt_cols(LoR, wname, mr);
    auto [Lo_of_HiR, Hi_of_HiR] = dwt_cols(HiR, wname, mr);

    return {std::move(Lo_of_LoR),   // cA = LL
            std::move(Hi_of_LoR),   // cH = LH (horizontal detail)
            std::move(Lo_of_HiR),   // cV = HL (vertical detail)
            std::move(Hi_of_HiR)};  // cD = HH (diagonal detail)
}

Value idwt2(const Value &cA, const Value &cH,
            const Value &cV, const Value &cD,
            const std::string &wname,
            long long outRows, long long outCols,
            std::pmr::memory_resource *mr)
{
    // Probed R2025b class contract: every numeric class (logical and the
    // integer types included) is accepted and computed in double; a
    // single anywhere makes the result single; complex coefficients
    // produce a complex result (the pipeline is linear, so the imaginary
    // part runs through the same double kernel). Sizes must agree.
    const Value *coefs[4] = {&cA, &cH, &cV, &cD};
    const char *names[4] = {"cA", "cH", "cV", "cD"};
    for (int k = 0; k < 4; ++k)
        internal::requireWaveClass(*coefs[k], "idwt2", names[k], k + 1,
                                   internal::WvClasses::AnyNumeric,
                                   /*allowComplex=*/true);
    for (int k = 1; k < 4; ++k) {
        if (coefs[k]->dims().rows() != cA.dims().rows()
            || coefs[k]->dims().cols() != cA.dims().cols())
            throw Error("idwt2: cA, cH, cV and cD must have the same size "
                        "(got " + internal::shapeOf(cA) + " and "
                        + internal::shapeOf(*coefs[k]) + ").",
                        0, 0, "idwt2", "", "nk:idwt2:size");
    }

    const bool allEmpty = cA.numel() == 0 && cH.numel() == 0
        && cV.numel() == 0 && cD.numel() == 0;
    if (allEmpty)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    bool anySngl = false, anyCplx = false;
    for (const Value *c : coefs) {
        anySngl |= c->type() == ValueType::SINGLE;
        anyCplx |= c->type() == ValueType::COMPLEX;
    }

    // Real / imaginary double views of the four coefficients (the
    // imaginary view of a REAL coefficient is zero, of course).
    auto viewOf = [&](bool imag, std::pmr::memory_resource *m) {
        Value v[4];
        for (int k = 0; k < 4; ++k) {
            const Value &src = *coefs[k];
            const size_t R = src.dims().rows(), C = src.dims().cols();
            v[k] = Value::matrix(R, C, ValueType::DOUBLE, m);
            double *d = v[k].doubleDataMut();
            if (!imag) {
                for (size_t i = 0; i < src.numel(); ++i)
                    d[i] = src.elemAsDouble(i);
            } else if (src.type() == ValueType::COMPLEX) {
                for (size_t i = 0; i < src.numel(); ++i)
                    d[i] = src.complexElem(i).imag();
            }
            // else: imag view of a real coefficient stays zero-filled.
        }
        return std::array<Value, 4>{std::move(v[0]), std::move(v[1]),
                                    std::move(v[2]), std::move(v[3])};
    };

    auto runPipeline = [&](const std::array<Value, 4> &v,
                           std::pmr::memory_resource *m) {
        size_t la_rows = v[0].dims().rows();
        size_t la_cols = v[0].dims().cols();
        size_t natRows = 0, natCols = 0;
        {
            auto a = Value::matrix(la_rows, 1, ValueType::DOUBLE, m);
            auto d = Value::matrix(la_rows, 1, ValueType::DOUBLE, m);
            Value probe = idwt(a, d, wname, /*len=*/-1, m);
            natRows = probe.numel();
        }
        {
            auto a = Value::matrix(la_cols, 1, ValueType::DOUBLE, m);
            auto d = Value::matrix(la_cols, 1, ValueType::DOUBLE, m);
            Value probe = idwt(a, d, wname, /*len=*/-1, m);
            natCols = probe.numel();
        }
        Value LoR = idwt_cols(v[0], v[1], wname, natRows, m);
        Value HiR = idwt_cols(v[2], v[3], wname, natRows, m);
        return idwt_rows(LoR, HiR, wname, natCols, m);
    };

    std::array<Value, 4> re = viewOf(false, mr);
    Value outR = runPipeline(re, mr);
    Value outI;
    if (anyCplx) {
        std::array<Value, 4> im = viewOf(true, mr);
        outI = runPipeline(im, mr);
    }

    // SIZ (probed R2025b): the reconstruction is NATURAL and SIZ is the
    // CENTRAL sub-matrix kept (idwt2(...,[2 3]) of a 4x4 natural yields
    // rows 2..3 x cols 1..3) — not a reconstruction target length.
    if (outRows >= 0 || outCols >= 0) {
        const std::size_t natR = outR.dims().rows();
        const std::size_t natC = outR.dims().cols();
        const std::size_t keepR = outRows >= 0
            ? static_cast<std::size_t>(outRows) : natR;
        const std::size_t keepC = outCols >= 0
            ? static_cast<std::size_t>(outCols) : natC;
        if (keepR > natR || keepC > natC)
            throw Error("idwt2: input 6 (SIZ) exceeds the natural "
                        "reconstruction size " + std::to_string(natR)
                        + "x" + std::to_string(natC) + ".",
                        0, 0, "idwt2", "", "nk:idwt2:size");
        const std::size_t r0 = (natR - keepR) / 2;
        const std::size_t c0 = (natC - keepC) / 2;
        auto crop = [&](const Value &src) {
            Value dst = Value::matrix(keepR, keepC, ValueType::DOUBLE, mr);
            for (std::size_t c = 0; c < keepC; ++c)
                for (std::size_t r = 0; r < keepR; ++r)
                    dst.doubleDataMut()[c * keepR + r]
                        = src.elemAsDouble((c0 + c) * natR + r0 + r);
            return dst;
        };
        outR = crop(outR);
        if (anyCplx) outI = crop(outI);
    }

    if (!anyCplx) {
        if (anySngl) {
            Value s = internal::allocLike(outR, ValueType::SINGLE, mr);
            for (std::size_t i = 0; i < outR.numel(); ++i)
                internal::storeElem(s, i, outR.elemAsDouble(i));
            return s;
        }
        return outR;
    }
    Value y = Value::complexMatrix(outR.dims().rows(), outR.dims().cols(), mr);
    Complex *yd = y.complexDataMut();
    for (std::size_t i = 0; i < outR.numel(); ++i)
        yd[i] = Complex(outR.elemAsDouble(i), outI.elemAsDouble(i));
    return y;
}

} // namespace numkit::wavelet
