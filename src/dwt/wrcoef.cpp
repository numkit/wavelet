// toolboxes/wavelet/src/dwt/wrcoef.cpp
//
// 1-D single-band reconstruction — wrcoef.
//
// MATLAB R2025b semantics (verified end-to-end via probe scripts):
//
//   y = wrcoef(type, c, l, wname[, n])
//
// The (c, l) pair is the output of `wavedec`. Layout (nk convention,
// matching the MATLAB doc):
//
//   c = [cA_N, cD_N, cD_{N-1}, ..., cD_1]
//   l = [|cA_N|, |cD_N|, ..., |cD_1|, |x|]            (length N+2)
//   N = length(l) - 2  (the number of decomposition levels)
//
//   * type ∈ {'a', 'd'} — reconstruct from approximation or detail.
//   * n is the level to keep:
//       'a': n ∈ [0, N], default = N.
//            n = 0 → full reconstruction (= original signal).
//            n ∈ [1, N] → keep cA_N and cD_{n+1}..cD_N, zero cD_1..cD_n,
//            then run waverec.
//       'd': n ∈ [1, N], default = N.
//            Zero cA_N and every cD_k except k = n; then waverec.
//   * wname is the wavelet name (haar, db1..db4, sym2/sym4, coif1).
//     The two-filter form `wrcoef(type, c, l, Lo_R, Hi_R[, n])` is
//     intentionally NOT supported in this release — pass wname instead.
//
// Output y is a row vector of length |x| (matches nk's wavedec/
// waverec which always emit row outputs). MATLAB preserves the
// orientation of c into y; nk's wavedec coerces c into a row
// internally, so wrcoef output is row-shaped here too.

#include <numkit/wavelet/dwt/multilevel.hpp>

// Compute-only TU: Value substrate + Error, no engine. The wrcoef builtin
// (CallContext wrapper) lives in dwt/wrcoef_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace numkit::wavelet {

namespace {

std::vector<double> vec_from_value(const Value &v)
{
    std::vector<double> out(v.numel());
    for (size_t i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

Value row_from_vec(const std::vector<double> &v,
                   std::pmr::memory_resource *mr)
{
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

std::string lower_type(std::string s)
{
    for (auto &ch : s)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

void wrcoef_argcheck(const std::string &tl, const Value &c, const Value &l, int n)
{
    if (tl != "a" && tl != "d")
        throw Error("wrcoef: type must be 'a' or 'd'",
                    0, 0, "wrcoef", "", "numkit:wrcoef:type");
    if (n < -1)
        throw Error("wrcoef: level n must be a non-negative integer",
                    0, 0, "wrcoef", "", "numkit:wrcoef:level");

    const size_t Lcount = l.numel();
    if (Lcount < 3)
        throw Error("wrcoef: l must have at least 3 entries",
                    0, 0, "wrcoef", "", "numkit:wrcoef:l");
    const int N = static_cast<int>(Lcount) - 2;
    const int effN = (n == -1) ? N : n;

    if (tl == "a") {
        if (effN < 0 || effN > N)
            throw Error("wrcoef: for type='a', level must satisfy 0 <= n <= "
                        + std::to_string(N),
                        0, 0, "wrcoef", "", "numkit:wrcoef:level");
    } else { // 'd'
        if (effN < 1 || effN > N)
            throw Error("wrcoef: for type='d', level must satisfy 1 <= n <= "
                        + std::to_string(N),
                        0, 0, "wrcoef", "", "numkit:wrcoef:level");
    }

    size_t expectedC = static_cast<size_t>(l.elemAsDouble(0));
    for (int k = 0; k < N; ++k)
        expectedC += static_cast<size_t>(l.elemAsDouble(1 + k));
    if (c.numel() != expectedC)
        throw Error("wrcoef: c length does not match l bookkeeping",
                    0, 0, "wrcoef", "", "numkit:wrcoef:cl");
}

} // anonymous

Value wrcoef(const std::string &type, const Value &c, const Value &l,
             const std::string &wname, int n,
             std::pmr::memory_resource *mr)
{
    const std::string tl = lower_type(type);
    wrcoef_argcheck(tl, c, l, n);

    const size_t Lcount = l.numel();
    const int N = static_cast<int>(Lcount) - 2;
    if (n == -1)
        n = N;            // sentinel: caller passed -1 to mean "default"

    auto cv = vec_from_value(c);
    auto sliceLen = [&](size_t idx) -> size_t {
        return static_cast<size_t>(l.elemAsDouble(idx));
    };

    // Detail at MATLAB level k (1-based, k=1 finest, k=N coarsest)
    // lives in c at L-index (1 + N - k). Compute the [start, end) byte
    // ranges in c for each detail.
    std::vector<std::pair<size_t, size_t>> dRange(N + 1);   // index 1..N
    {
        size_t off = sliceLen(0);
        for (int idxL = 1; idxL <= N; ++idxL) {
            const size_t dlen = sliceLen(idxL);
            const int level   = N - idxL + 1;     // L-pos idxL ↔ level
            dRange[level] = {off, off + dlen};
            off += dlen;
        }
    }

    if (tl == "a") {
        // Zero detail levels 1..n  (do nothing if n==0).
        for (int k = 1; k <= n; ++k) {
            for (size_t i = dRange[k].first; i < dRange[k].second; ++i)
                cv[i] = 0.0;
        }
    } else {
        // Zero cA_N.
        for (size_t i = 0; i < sliceLen(0); ++i) cv[i] = 0.0;
        // Zero every detail except level n.
        for (int k = 1; k <= N; ++k) {
            if (k == n) continue;
            for (size_t i = dRange[k].first; i < dRange[k].second; ++i)
                cv[i] = 0.0;
        }
    }

    Value cMod = row_from_vec(cv, mr);
    return waverec(cMod, l, wname, mr);
}

} // namespace numkit::wavelet
