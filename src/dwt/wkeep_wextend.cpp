// toolboxes/wavelet/src/dwt/wkeep_wextend.cpp
//
// Vector-form wkeep and wextend (Wavelet Toolbox boundary helpers).
//
// Contract probed against MATLAB R2025b on 2026-09-18
// (build/probe41/44/49/50/52): both helpers are class-preserving —
// wkeep accepts every array class including char and complex, wextend
// takes double/single/integer (logical and char rejected) with complex
// allowed; an invalid LEN makes wkeep return X UNCHANGED (not an
// error); wextend's LOC letters map l/u → start, r/d → end, b → both,
// n → no extension, anything else → both.

#include <numkit/wavelet/dwt/wkeep_wextend.hpp>
#include <numkit/wavelet/internal/type_gates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The wkeep /
// wextend builtins (CallContext wrappers) live in dwt/wkeep_wextend_reg.cpp. The
// 1-D extension core detail::extend1D stays HERE — it is a compute helper
// invoked by the public wextend(), not a register-side parser.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace numkit::wavelet {

namespace {

void readShape(const Value &x, size_t &rows, size_t &cols)
{
    rows = x.dims().rows();
    cols = x.dims().cols();
    if (rows == 0 && cols == 0) { rows = 1; cols = 0; }
}

bool isCol(size_t rows, size_t cols)
{
    return cols == 1 && rows > 1;
}

void outShape(bool col, size_t outN,
              size_t &outRows, size_t &outCols)
{
    if (col) { outRows = outN; outCols = 1; }
    else     { outRows = 1;    outCols = outN; }
}

std::string lower(std::string s)
{
    for (auto &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// A whole LEN value that is a nonnegative integer in [0, N]; the probed
// reference returns X UNCHANGED for anything else (negative, too big,
// non-integer) instead of erroring.
bool validKeepLen(double n, long long N)
{
    return n >= 0.0 && n <= static_cast<double>(N) && n == std::floor(n);
}

} // anonymous

// ── Public C++ API (see dwt/wkeep_wextend.hpp) ────────────────────────

Value wkeep(const Value &x, const Value &len, const Value &opt,
            std::pmr::memory_resource *mr)
{
    // Class gate: every array class passes through (char included — the
    // reference keeps it); cells/structs/strings reject. 3-D inputs are
    // outside the documented 1-D/2-D surface.
    internal::requireWaveClass(x, "wkeep", "X", 1,
                               internal::WvClasses::AnyWithChar,
                               /*allowComplex=*/true);
    if (x.dims().ndim() > 2)
        throw Error("wkeep: X must be a vector or a 2-D matrix (got a "
                        + internal::shapeOf(x) + " array).",
                    0, 0, "wkeep", "", "numkit:wkeep:inputType");
    if (len.isChar() || len.isString())
        throw Error("wkeep: input 2 (L) must be numeric (got a "
                        + std::string(mtypeName(len.type())) + " array).",
                    0, 0, "wkeep", "", "numkit:wkeep:inputType");

    size_t rows, cols;
    readShape(x, rows, cols);
    const size_t N = rows * cols;
    const bool xIsVector = (rows == 1) || (cols == 1);

    if (len.numel() == 2) {
        // 2-element LEN: for a VECTOR x only the element matching the
        // non-singleton dimension applies (R for a column, C for a row —
        // the other is ignored entirely); for a matrix it is the [R C]
        // sub-matrix form. Invalid values return x unchanged.
        const double l0 = len.elemAsDouble(0);
        const double l1 = len.elemAsDouble(1);
        long long R, C;
        if (xIsVector) {
            if (cols > 1 || (rows == 1 && cols == 1)) {   // row-like
                if (!validKeepLen(l1, static_cast<long long>(N)))
                    return internal::copyInClass(x, mr);
                R = 1; C = static_cast<long long>(l1);
            } else {                                       // column
                if (!validKeepLen(l0, static_cast<long long>(N)))
                    return internal::copyInClass(x, mr);
                R = static_cast<long long>(l0); C = 1;
            }
        } else {
            if (!validKeepLen(l0, static_cast<long long>(rows))
                || !validKeepLen(l1, static_cast<long long>(cols)))
                return internal::copyInClass(x, mr);
            R = static_cast<long long>(l0);
            C = static_cast<long long>(l1);
        }
        if (R < 0 || C < 0)
            return internal::copyInClass(x, mr);

        long long fr1, fc1;  // 1-based corners
        if (!opt.isEmpty() && opt.numel() == 2) {
            fr1 = static_cast<long long>(opt.elemAsDouble(0));
            fc1 = static_cast<long long>(opt.elemAsDouble(1));
        } else if (!opt.isEmpty() && !opt.isChar() && !opt.isString()) {
            // A scalar FIRST on a 2-element L: the reference bails with
            // its generic invalid-argument diagnostic.
            throw Error("wkeep: invalid optional argument FIRST (expected "
                        "'c', 'l', 'r' or a [firstRow firstCol] pair).",
                        0, 0, "wkeep", "", "numkit:wkeep:opt");
        } else {
            // Central — same formula as 1-D: floor((N - n) / 2) + 1.
            fr1 = static_cast<long long>((rows - static_cast<size_t>(R)) / 2) + 1;
            fc1 = static_cast<long long>((cols - static_cast<size_t>(C)) / 2) + 1;
        }
        if (fr1 < 1 || fc1 < 1 ||
            fr1 + R - 1 > static_cast<long long>(rows) ||
            fc1 + C - 1 > static_cast<long long>(cols))
            throw Error("wkeep: requested window is out of range.",
                        0, 0, "wkeep", "", "numkit:wkeep:range");

        const bool cplx = x.type() == ValueType::COMPLEX;
        Value y = cplx ? Value::complexMatrix(static_cast<size_t>(R),
                                              static_cast<size_t>(C), mr)
                       : Value::matrix(static_cast<size_t>(R),
                                       static_cast<size_t>(C), x.type(), mr);
        if (R == 0 || C == 0) return y;
        for (long long c = 0; c < C; ++c) {
            const size_t srcC = static_cast<size_t>(fc1 - 1 + c);
            for (long long r = 0; r < R; ++r) {
                const size_t srcR = static_cast<size_t>(fr1 - 1 + r);
                const size_t src = srcC * rows + srcR;
                const size_t dst = static_cast<size_t>(c) * static_cast<size_t>(R)
                                 + static_cast<size_t>(r);
                if (cplx) y.complexDataMut()[dst] = x.complexElem(src);
                else internal::storeElem(y, dst, x.elemAsDouble(src));
            }
        }
        return y;
    }

    // 1-D form: scalar LEN.
    if (len.numel() != 1)
        throw Error("wkeep: input 2 (L) must be a scalar length or a "
                        "[R C] pair (got a " + internal::shapeOf(len)
                        + " array).",
                    0, 0, "wkeep", "", "numkit:wkeep:inputType");
    const double nD = len.elemAsDouble(0);
    if (!validKeepLen(nD, static_cast<long long>(N)))
        return internal::copyInClass(x, mr);
    const long long n = static_cast<long long>(nD);

    long long start1 = 1;                                  // 1-based start
    if (!opt.isEmpty()) {
        if (opt.isChar() || opt.isString()) {
            const std::string s = lower(opt.toString());
            if (s == "c" || s == "central" || s == "centered" || s == "centred")
                start1 = static_cast<long long>((N - static_cast<size_t>(n)) / 2) + 1;
            else if (s == "l")
                start1 = 1;
            else if (s == "r")
                start1 = static_cast<long long>(N) - n + 1;
            else
                throw Error("wkeep: unknown option '" + s +
                            "' (expected 'c', 'l', 'r' or a numeric start)",
                            0, 0, "wkeep", "", "numkit:wkeep:opt");
        } else {
            const double f = opt.elemAsDouble(0);
            if (f != std::floor(f))
                throw Error("wkeep: optional argument FIRST must be an "
                            "integer start index.",
                            0, 0, "wkeep", "", "numkit:wkeep:opt");
            start1 = static_cast<long long>(f);
        }
    } else {
        start1 = static_cast<long long>((N - static_cast<size_t>(n)) / 2) + 1;
    }
    if (start1 < 1 || start1 + n - 1 > static_cast<long long>(N))
        throw Error("wkeep: requested window is out of range.",
                    0, 0, "wkeep", "", "numkit:wkeep:range");

    const bool col = isCol(rows, cols);
    size_t outRows, outCols;
    outShape(col, static_cast<size_t>(n), outRows, outCols);
    const bool cplx = x.type() == ValueType::COMPLEX;
    Value y = cplx ? Value::complexMatrix(outRows, outCols, mr)
                   : Value::matrix(outRows, outCols, x.type(), mr);
    if (n == 0) return y;
    for (long long k = 0; k < n; ++k) {
        const size_t src = static_cast<size_t>(start1 - 1 + k);
        if (cplx) y.complexDataMut()[k] = x.complexElem(src);
        else internal::storeElem(y, static_cast<size_t>(k), x.elemAsDouble(src));
    }
    return y;
}

namespace detail {

// 1-D extension core. Takes a length-N source `xv` and produces an output
// vector with `leftN` left-pad samples + the (possibly edge-padded for
// 'per') middle + `rightN` right-pad samples.
//
// Modes (verified vs MATLAB R2025b on x = [1 2 3 4 5], lf = 2):
//   'sym' / 'symh' — half-point symmetric (edge sample replicated)
//                    pre = [x(lf), …, x(1)],   post = [x(N), …, x(N-lf+1)]
//                    [2 1 | 1 2 3 4 5 | 5 4]
//   'symw'         — whole-point symmetric (edge NOT replicated)
//                    pre = [x(lf+1), …, x(2)], post = [x(N-1), …, x(N-lf)]
//                    [3 2 | 1 2 3 4 5 | 4 3]
//   'asym' / 'asymh' — half-point antisymmetric
//                    pre = -[x(lf), …, x(1)],  post = -[x(N), …, x(N-lf+1)]
//                    [-2 -1 | 1 2 3 4 5 | -5 -4]
//   'asymw'        — whole-point antisymmetric (reflect through endpoint)
//                    pre[i]  = 2·x(1) - x(lf-i+1)  reversed → 2·x(1) - x(j)
//                    [-1 0 | 1 2 3 4 5 | 6 7]   (linear-like for x=1..5)
//   'sp0'          — order-0 spline = replicate edge sample
//                    [1 1 | 1..5 | 5 5]
//   'sp1' / 'spd'  — order-1 spline = linear extrapolation ('spd' is the
//                    reference's alias for the same values, probed R2025b)
//                    pre[i]  = x(1) - (lf-i)·(x(2) - x(1))   for i=0..lf-1
//                    post[i] = x(N) + (i+1)·(x(N) - x(N-1))
//                    [-1 0 | 1..5 | 6 7]
//   'per'          — periodic with edge-pad on odd N. If N odd, MATLAB pads
//                    x → x' = [x x(end)]; pre = last lf of x', post = first lf.
//   'zpd'          — zero pad
//   'ppd'          — pure periodic: pre = last lf of x; post = first lf
//
// Element type is templated: the class-preserving wextend runs the same
// permutation/linear-combination core on double and Complex buffers.
template <typename T>
static std::vector<T> extend1D(const std::vector<T> &xv,
                               long long lf, char side,
                               const std::string &mode)
{
    const long long N = static_cast<long long>(xv.size());
    if (lf < 0) lf = 0;
    // For 'per' the data segment is the (possibly edge-padded) version of x.
    std::vector<T> src = xv;
    if (mode == "per" && (N % 2) == 1 && N > 0) src.push_back(xv.back());
    const long long M = static_cast<long long>(src.size());

    auto sampleLeft = [&](long long k) -> T {
        // k = 0..lf-1, with k=0 = furthest from data (output[0]).
        if (mode == "sym" || mode == "symh") {
            return xv[static_cast<size_t>(lf - 1 - k)];
        }
        if (mode == "symw") {
            // pre = [x(lf+1), …, x(2)] in 1-based → indices [lf, lf-1, …, 1] 0-based.
            // For output position k (0..lf-1), src idx = lf - k.
            const long long idx = lf - k;
            return (idx < N) ? xv[static_cast<size_t>(idx)] : xv.back();
        }
        if (mode == "asym" || mode == "asymh") {
            return -xv[static_cast<size_t>(lf - 1 - k)];
        }
        if (mode == "asymw") {
            // 2·x(1) - x(j) where j is the symw index (1-based j = lf+1-k = lf-k+1).
            // 0-based: idx = lf - k.
            const long long idx = lf - k;
            const T xj = (idx < N) ? xv[static_cast<size_t>(idx)] : xv.back();
            return T(2) * xv[0] - xj;
        }
        if (mode == "sp0") {
            return xv[0];
        }
        if (mode == "sp1" || mode == "spd") {
            // Linear extrapolation using slope x(2) - x(1).
            const T slope = (N >= 2) ? (xv[1] - xv[0]) : T(0);
            return xv[0] - T(lf - k) * slope;
        }
        if (mode == "per") {
            return src[static_cast<size_t>(M - lf + k)];
        }
        if (mode == "ppd") {
            return xv[static_cast<size_t>(N - lf + k)];
        }
        return T(0);  // zpd
    };
    auto sampleRight = [&](long long k) -> T {
        // k = 0..lf-1, k=0 = closest to data.
        if (mode == "sym" || mode == "symh") {
            return xv[static_cast<size_t>(N - 1 - k)];
        }
        if (mode == "symw") {
            // post = [x(N-1), …, x(N-lf)] in 1-based → indices [N-2, N-3, …, N-lf-1] 0-based.
            const long long idx = N - 2 - k;
            return (idx >= 0) ? xv[static_cast<size_t>(idx)] : xv.front();
        }
        if (mode == "asym" || mode == "asymh") {
            return -xv[static_cast<size_t>(N - 1 - k)];
        }
        if (mode == "asymw") {
            const long long idx = N - 2 - k;
            const T xj = (idx >= 0) ? xv[static_cast<size_t>(idx)] : xv.front();
            return T(2) * xv[static_cast<size_t>(N - 1)] - xj;
        }
        if (mode == "sp0") {
            return xv[static_cast<size_t>(N - 1)];
        }
        if (mode == "sp1" || mode == "spd") {
            const T slope = (N >= 2) ? (xv[static_cast<size_t>(N - 1)]
                                        - xv[static_cast<size_t>(N - 2)]) : T(0);
            return xv[static_cast<size_t>(N - 1)] + T(k + 1) * slope;
        }
        if (mode == "per") {
            return src[static_cast<size_t>(k)];
        }
        if (mode == "ppd") {
            return xv[static_cast<size_t>(k)];
        }
        return T(0);
    };

    const long long leftN  = (side == 'r') ? 0 : lf;
    const long long rightN = (side == 'l') ? 0 : lf;
    const std::vector<T> &mid = (mode == "per") ? src : xv;
    const long long Mlen = static_cast<long long>(mid.size());
    std::vector<T> out;
    out.reserve(static_cast<size_t>(leftN + Mlen + rightN));
    for (long long i = 0; i < leftN; ++i) out.push_back(sampleLeft(i));
    for (long long i = 0; i < Mlen;  ++i) out.push_back(mid[static_cast<size_t>(i)]);
    for (long long i = 0; i < rightN; ++i) out.push_back(sampleRight(i));
    return out;
}

// Explicit instantiations for the two buffer flavours wextend uses.
template std::vector<double> extend1D<double>(const std::vector<double> &,
                                              long long, char, const std::string &);
template std::vector<Complex> extend1D<Complex>(const std::vector<Complex> &,
                                                long long, char, const std::string &);

} // namespace detail

namespace {

// Probed R2025b LOC semantics: 'l'/'u' extend at the START of an axis,
// 'r'/'d' at the END, 'b' both, 'n' none; anything unrecognized (numbers
// included) behaves as 'b'. One letter applies to every extended axis,
// two letters give (row-axis, column-axis) for the 2-D forms.
char parseLocChar(char c)
{
    if (c == 'l' || c == 'u') return 'l';
    if (c == 'r' || c == 'd') return 'r';
    if (c == 'n') return 'n';
    return 'b';
}

// Read LF: a scalar nonnegative integer, or a [rows cols] pair for the
// 2-D forms. Non-integer / complex / negative values reject.
void parseExtLf(const Value &lf, const char *fn,
                long long &lfRows, long long &lfCols, bool twoDim)
{
    if (lf.type() == ValueType::COMPLEX)
        throw Error(std::string(fn) + ": input 4 (L) must be a nonnegative "
                    "integer (got a complex value).",
                    0, 0, fn, "", std::string("numkit:") + fn + ":expectedInteger");
    if (lf.isCell() || lf.isStruct() || lf.isChar() || lf.isString())
        throw Error(std::string(fn) + ": input 4 (L) must be numeric.",
                    0, 0, fn, "", std::string("numkit:") + fn + ":expectedInteger");
    if (lf.numel() == 2 && twoDim) {
        const double a = lf.elemAsDouble(0);
        const double b = lf.elemAsDouble(1);
        if (a < 0 || b < 0)
            throw Error(std::string(fn) + ": input 4 (L) must be nonnegative.",
                        0, 0, fn, "", std::string("numkit:") + fn + ":expectedNonnegative");
        if (a != std::floor(a) || b != std::floor(b))
            throw Error(std::string(fn) + ": input 4 (L) must be an integer.",
                        0, 0, fn, "", std::string("numkit:") + fn + ":expectedInteger");
        lfRows = static_cast<long long>(a);
        lfCols = static_cast<long long>(b);
        return;
    }
    if (lf.numel() != 1)
        throw Error(std::string(fn) + ": input 4 (L) must be a scalar "
                    "nonnegative integer.",
                    0, 0, fn, "", std::string("numkit:") + fn + ":expectedInteger");
    const double v = lf.elemAsDouble(0);
    if (v < 0)
        throw Error(std::string(fn) + ": input 4 (L) must be nonnegative.",
                    0, 0, fn, "", std::string("numkit:") + fn + ":expectedNonnegative");
    if (v != std::floor(v))
        throw Error(std::string(fn) + ": input 4 (L) must be an integer.",
                    0, 0, fn, "", std::string("numkit:") + fn + ":expectedInteger");
    lfRows = lfCols = static_cast<long long>(v);
}

} // anonymous

// ── Public C++ API: wextend (see dwt/wkeep_wextend.hpp) ───────────────
// Reuses detail::extend1D (the file-internal 1-D extension core).
Value wextend(const Value &type, const std::string &modeRaw, const Value &x,
              const Value &lf, const std::string &sideRaw,
              std::pmr::memory_resource *mr)
{
    // type: 1, 2, '1', '1d', '2', '2d', 'ar', 'ac' (probed R2025b; the
    // reference also takes numeric 1/2 and the 'd'-suffixed aliases).
    int dim = 0;
    bool extRows = true;
    bool extCols = true;
    if (type.isChar() || type.isString()) {
        const std::string s = lower(type.toString());
        if (s == "1" || s == "1d") dim = 1;
        else if (s == "2" || s == "2d") dim = 2;
        else if (s == "ar") { dim = 2; extRows = true;  extCols = false; }
        else if (s == "ac") { dim = 2; extRows = false; extCols = true; }
        else
            throw Error("wextend: TYPE must be 1, '1d', 2, '2d', 'ar' or 'ac'",
                        0, 0, "wextend", "", "numkit:wextend:dim");
    } else if (type.numel() == 1 && !type.isCell() && !type.isStruct()) {
        const int t = static_cast<int>(type.elemAsDouble(0));
        if (t != 1 && t != 2)
            throw Error("wextend: TYPE must be 1, '1d', 2, '2d', 'ar' or 'ac'",
                        0, 0, "wextend", "", "numkit:wextend:dim");
        dim = t;
    } else {
        throw Error("wextend: TYPE must be 1, '1d', 2, '2d', 'ar' or 'ac'",
                    0, 0, "wextend", "", "numkit:wextend:dim");
    }

    // Class gate: double / single / integer classes, complex allowed;
    // logical and char are rejected by the reference (invalidType).
    internal::requireWaveClass(x, "wextend", "X", 3,
                               internal::WvClasses::IntWavelet,
                               /*allowComplex=*/true);
    if (x.dims().ndim() > 2)
        throw Error("wextend: X must be a vector or a 2-D matrix (got a "
                        + internal::shapeOf(x) + " array).",
                    0, 0, "wextend", "", "numkit:wextend:inputType");

    const std::string mode = lower(modeRaw);
    static const char *kModes[] = {"sym", "symh", "symw",
                                    "asym", "asymh", "asymw",
                                    "sp0", "sp1", "spd",
                                    "per", "zpd", "ppd"};
    bool modeOK = false;
    for (const char *m : kModes) if (mode == m) { modeOK = true; break; }
    if (!modeOK)
        throw Error("wextend: unsupported mode '" + mode +
                    "' (supported: sym/symh/symw, asym/asymh/asymw, "
                    "sp0/sp1/spd, per, zpd, ppd)",
                    0, 0, "wextend", "", "numkit:wextend:mode");

    long long lfRows = 0, lfCols = 0;
    parseExtLf(lf, "wextend", lfRows, lfCols, dim == 2);

    // LOC: per-axis side letters (see parseLocChar). 'n' → x unchanged.
    char sideRows = 'b', sideCols = 'b';
    {
        const std::string s = lower(sideRaw);
        if (s.empty()) {
            // no LOC argument
        } else if (s.size() >= 2 && dim == 2) {
            sideRows = parseLocChar(s[0]);
            sideCols = parseLocChar(s[1]);
        } else if (s.size() >= 1) {
            sideRows = sideCols = parseLocChar(s[0]);
        }
    }
    if (sideRows == 'n' && sideCols == 'n')
        return internal::copyInClass(x, mr);
    if (dim == 1) {
        // A single axis: the LOC letter addresses it directly.
        const char side = sideRows;               // 'l'/'r'/'b' ('n' handled)
        size_t rows, cols;
        readShape(x, rows, cols);
        const size_t N = rows * cols;
        const bool col = isCol(rows, cols);
        const bool cplx = x.type() == ValueType::COMPLEX;
        size_t outN = 0;
        Value y;
        if (cplx) {
            std::vector<Complex> xv(N);
            for (size_t i = 0; i < N; ++i) xv[i] = x.complexElem(i);
            std::vector<Complex> ext = detail::extend1D(xv, lfRows, side, mode);
            outN = ext.size();
            size_t outRows, outCols;
            outShape(col, outN, outRows, outCols);
            y = Value::complexMatrix(outRows, outCols, mr);
            std::copy(ext.begin(), ext.end(), y.complexDataMut());
            return y;
        }
        std::vector<double> xv(N);
        for (size_t i = 0; i < N; ++i) xv[i] = x.elemAsDouble(i);
        std::vector<double> ext = detail::extend1D(xv, lfRows, side, mode);
        outN = ext.size();
        size_t outRows, outCols;
        outShape(col, outN, outRows, outCols);
        y = Value::matrix(outRows, outCols, x.type(), mr);
        for (size_t i = 0; i < outN; ++i)
            internal::storeElem(y, i, ext[i]);
        return y;
    }

    // 2-D matrix path (type=2 / '2d' / 'ar' / 'ac'). Work on a row-major
    // buffer of the input's element flavour, extend each axis in turn,
    // materialise once in the input's class.
    size_t rows, cols;
    readShape(x, rows, cols);
    const bool cplx = x.type() == ValueType::COMPLEX;
    const char sideR = sideRows == 'n' ? 0 : sideRows;
    const char sideC = sideCols == 'n' ? 0 : sideCols;
    const long long lfR = sideR ? lfRows : 0;
    const long long lfC = sideC ? lfCols : 0;

    if (!cplx) {
        std::vector<std::vector<double>> data(rows, std::vector<double>(cols));
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                data[r][c] = x.elemAsDouble(c * rows + r);

        if (extRows && lfR > 0) {
            std::vector<std::vector<double>> ext_col;
            ext_col.reserve(cols);
            for (size_t c = 0; c < cols; ++c) {
                std::vector<double> col_v(rows);
                for (size_t r = 0; r < rows; ++r) col_v[r] = data[r][c];
                ext_col.push_back(detail::extend1D(col_v, lfR, sideR, mode));
            }
            const size_t newRows = ext_col[0].size();
            data.assign(newRows, std::vector<double>(cols));
            for (size_t c = 0; c < cols; ++c)
                for (size_t r = 0; r < newRows; ++r)
                    data[r][c] = ext_col[c][r];
            rows = newRows;
        }
        if (extCols && lfC > 0) {
            std::vector<std::vector<double>> ext_row;
            ext_row.reserve(rows);
            for (size_t r = 0; r < rows; ++r)
                ext_row.push_back(detail::extend1D(data[r], lfC, sideC, mode));
            const size_t newCols = ext_row[0].size();
            for (auto &row : data) row.assign(newCols, 0.0);
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < newCols; ++c)
                    data[r][c] = ext_row[r][c];
            cols = newCols;
        }

        Value y = Value::matrix(rows, cols, x.type(), mr);
        for (size_t c = 0; c < cols; ++c)
            for (size_t r = 0; r < rows; ++r)
                internal::storeElem(y, c * rows + r, data[r][c]);
        return y;
    }

    std::vector<std::vector<Complex>> data(rows, std::vector<Complex>(cols));
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            data[r][c] = x.complexElem(c * rows + r);
    if (extRows && lfR > 0) {
        std::vector<std::vector<Complex>> ext_col;
        ext_col.reserve(cols);
        for (size_t c = 0; c < cols; ++c) {
            std::vector<Complex> col_v(rows);
            for (size_t r = 0; r < rows; ++r) col_v[r] = data[r][c];
            ext_col.push_back(detail::extend1D(col_v, lfR, sideR, mode));
        }
        const size_t newRows = ext_col[0].size();
        data.assign(newRows, std::vector<Complex>(cols));
        for (size_t c = 0; c < cols; ++c)
            for (size_t r = 0; r < newRows; ++r)
                data[r][c] = ext_col[c][r];
        rows = newRows;
    }
    if (extCols && lfC > 0) {
        std::vector<std::vector<Complex>> ext_row;
        ext_row.reserve(rows);
        for (size_t r = 0; r < rows; ++r)
            ext_row.push_back(detail::extend1D(data[r], lfC, sideC, mode));
        const size_t newCols = ext_row[0].size();
        for (auto &row : data) row.assign(newCols, Complex(0.0, 0.0));
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < newCols; ++c)
                data[r][c] = ext_row[r][c];
        cols = newCols;
    }
    Value y = Value::complexMatrix(rows, cols, mr);
    for (size_t c = 0; c < cols; ++c)
        for (size_t r = 0; r < rows; ++r)
            y.complexDataMut()[c * rows + r] = data[r][c];
    return y;
}

} // namespace numkit::wavelet
