// toolboxes/wavelet/src/dwt/multilevel.cpp
//
// Multi-level DWT (wavedec) and reconstruction (waverec) wrap the
// single-level dwt / idwt primitives. The (C, L) layout is the
// MATLAB-canonical packing:
//
//   C = [cA_n, cD_n, cD_{n-1}, ..., cD_1]    (concatenated row vector)
//   L = [|cA_n|, |cD_n|, |cD_{n-1}|, ..., |cD_1|, |x|]
//
// (See MATLAB Wavelet Toolbox `wavedec` doc.) The bookkeeping is
// fragile but very small — `appcoef` / `detcoef` just slice C using
// the cumulative offsets implied by L.

#include <numkit/wavelet/dwt/multilevel.hpp>
#include <numkit/wavelet/dwt/dwt.hpp>
#include <numkit/wavelet/denoise/denoise.hpp>
#include <numkit/wavelet/filter/wfilters.hpp>

// Compute-only TU: Value substrate + Error, no engine. The wavedec /
// waverec / appcoef / detcoef builtins (CallContext wrappers) live in
// dwt/multilevel_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace numkit::wavelet {

namespace {

Value rowFromVec(const std::vector<double> &v,
                 std::pmr::memory_resource *mr) {
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

std::vector<double> vecFromValue(const Value &v) {
    std::vector<double> out(v.numel());
    for (size_t i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

} // anonymous

std::pair<Value, Value>
wavedec(const Value &x, int n, const std::string &wname,
        std::pmr::memory_resource *mr)
{
    if (n < 1)
        throw Error("wavedec: level must be ≥ 1",
                    0, 0, "wavedec", "", "numkit:wavedec:level");

    // Run n successive single-level DWTs on the running approximation.
    // Stash each cD in `details[level-1]` (0-indexed: details[0] = cD_1
    // (finest), details[n-1] = cD_n (coarsest)).
    std::vector<std::vector<double>> details(n);
    Value running = Value::matrix(1, x.numel(), ValueType::DOUBLE, mr);
    {
        double *rd = running.doubleDataMut();
        for (size_t i = 0; i < x.numel(); ++i) rd[i] = x.elemAsDouble(i);
    }

    Value finalApprox;
    for (int k = 0; k < n; ++k) {
        auto [cA, cD] = dwt(running, wname, mr);
        details[k] = vecFromValue(cD);
        running = cA;
        if (k == n - 1) finalApprox = cA;
    }
    auto approx = vecFromValue(finalApprox);

    // L = [|cA_n|, |cD_n|, |cD_{n-1}|, ..., |cD_1|, |x|]
    std::vector<double> L;
    L.reserve(n + 2);
    L.push_back(static_cast<double>(approx.size()));
    for (int k = n - 1; k >= 0; --k)
        L.push_back(static_cast<double>(details[k].size()));
    L.push_back(static_cast<double>(x.numel()));

    // C = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    size_t total = approx.size();
    for (auto &d : details) total += d.size();
    std::vector<double> C;
    C.reserve(total);
    C.insert(C.end(), approx.begin(), approx.end());
    for (int k = n - 1; k >= 0; --k)
        C.insert(C.end(), details[k].begin(), details[k].end());

    return {rowFromVec(C, mr), rowFromVec(L, mr)};
}

Value waverec(const Value &C, const Value &L, const std::string &wname,
              std::pmr::memory_resource *mr)
{
    const size_t Lcount = L.numel();
    if (Lcount < 3)
        throw Error("waverec: L must have at least 3 entries (1 level)",
                    0, 0, "waverec", "", "numkit:waverec:size");
    const int n = static_cast<int>(Lcount) - 2; // number of decomposition levels
    auto sliceLen = [&](size_t idx) -> size_t {
        return static_cast<size_t>(L.elemAsDouble(idx));
    };
    auto Cv = vecFromValue(C);

    // Approximation cA_n is the first L[0] entries of C.
    size_t off = 0;
    const size_t aLen = sliceLen(0);
    if (off + aLen > Cv.size())
        throw Error("waverec: C/L mismatch (approx)",
                    0, 0, "waverec", "", "numkit:waverec:bounds");
    std::vector<double> running(Cv.begin() + off, Cv.begin() + off + aLen);
    off += aLen;

    // Walk through details from coarsest (L[1]) to finest (L[n]).
    for (int k = 0; k < n; ++k) {
        const size_t dLen = sliceLen(1 + k);
        if (off + dLen > Cv.size())
            throw Error("waverec: C/L mismatch (detail)",
                        0, 0, "waverec", "", "numkit:waverec:bounds");
        std::vector<double> detail(Cv.begin() + off, Cv.begin() + off + dLen);
        off += dLen;

        // Target reconstruction length: L[2+k] (the next coarser
        // approximation length, or the original signal length at the
        // last step).
        const size_t recLen = sliceLen(2 + k);

        // Pack as Value rows for the idwt API.
        Value cA = rowFromVec(running, mr);
        Value cD = rowFromVec(detail, mr);
        Value next = idwt(cA, cD, wname, static_cast<long long>(recLen), mr);
        running = vecFromValue(next);
    }
    return rowFromVec(running, mr);
}

// Internal: appcoef with explicit Lo_R / Hi_R synthesis filters.
static Value appcoef_with_filters(const Value &C, const Value &L,
                                  const std::vector<double> &Lo_R,
                                  const std::vector<double> &Hi_R,
                                  int level,
                                  std::pmr::memory_resource *mr)
{
    const size_t Lcount = L.numel();
    if (Lcount < 3)
        throw Error("appcoef: L too short",
                    0, 0, "appcoef", "", "numkit:appcoef:size");
    const int nMax = static_cast<int>(Lcount) - 2;
    if (level < 0) level = nMax;          // default = coarsest
    if (level < 0 || level > nMax)
        throw Error("appcoef: level out of range",
                    0, 0, "appcoef", "", "numkit:appcoef:level");

    auto Cv = vecFromValue(C);
    auto sliceLen = [&](size_t idx) -> size_t {
        return static_cast<size_t>(L.elemAsDouble(idx));
    };

    std::vector<double> running(Cv.begin(), Cv.begin() + sliceLen(0));
    if (level == nMax) return rowFromVec(running, mr);

    size_t off = sliceLen(0);
    for (int k = 0; k < nMax - level; ++k) {
        const size_t dLen = sliceLen(1 + k);
        std::vector<double> detail(Cv.begin() + off, Cv.begin() + off + dLen);
        off += dLen;
        const size_t recLen = sliceLen(2 + k);
        Value cA = rowFromVec(running, mr);
        Value cD = rowFromVec(detail, mr);
        Value next = idwt_with_filters_pub(cA, cD, Lo_R, Hi_R,
                                           static_cast<long long>(recLen), mr);
        running = vecFromValue(next);
    }
    return rowFromVec(running, mr);
}

Value appcoef(const Value &C, const Value &L, const std::string &wname,
              int level,
              std::pmr::memory_resource *mr)
{
    auto fb = wavelet_filters(wname);
    return appcoef_with_filters(C, L, fb.Lo_R, fb.Hi_R, level, mr);
}

// Cross-TU access for the custom-filter appcoef form (used by
// multilevel_reg.cpp).
Value appcoef_with_filters_pub(const Value &C, const Value &L,
                               const std::vector<double> &Lo_R,
                               const std::vector<double> &Hi_R,
                               int level,
                               std::pmr::memory_resource *mr)
{
    return appcoef_with_filters(C, L, Lo_R, Hi_R, level, mr);
}

Value detcoef(const Value &C, const Value &L, int level,
              std::pmr::memory_resource *mr)
{
    const size_t Lcount = L.numel();
    if (Lcount < 3)
        throw Error("detcoef: L too short",
                    0, 0, "detcoef", "", "numkit:detcoef:size");
    const int nMax = static_cast<int>(Lcount) - 2;
    if (level < 1 || level > nMax)
        throw Error("detcoef: level out of range",
                    0, 0, "detcoef", "", "numkit:detcoef:level");

    // C layout: [cA_n, cD_n, cD_{n-1}, ..., cD_1].
    // Detail at MATLAB level `level` (1=finest, nMax=coarsest) lives
    // at C-index that we walk to from the front:
    //   skip cA_n  : L[0]
    //   skip cD_n, cD_{n-1}, ..., cD_{level+1}  : sum L[1..nMax-level]
    //   then dLen = L[nMax - level + 1]
    auto sliceLen = [&](size_t idx) -> size_t {
        return static_cast<size_t>(L.elemAsDouble(idx));
    };
    size_t off = sliceLen(0);
    for (int k = 0; k < nMax - level; ++k) off += sliceLen(1 + k);
    const size_t dLen = sliceLen(static_cast<size_t>(nMax - level + 1));

    auto Cv = vecFromValue(C);
    if (off + dLen > Cv.size())
        throw Error("detcoef: C/L bounds",
                    0, 0, "detcoef", "", "numkit:detcoef:bounds");
    std::vector<double> out(Cv.begin() + off, Cv.begin() + off + dLen);
    return rowFromVec(out, mr);
}

WenergyResult wenergy(const Value &C, const Value &L,
                      std::pmr::memory_resource *mr)
{
    auto Cv = vecFromValue(C);
    auto Lv = vecFromValue(L);
    if (Lv.size() < 3)
        throw Error("wenergy: L must have at least 3 entries (a + 1 detail + length)",
                    0, 0, "wenergy", "", "numkit:wenergy:L");
    const size_t nLev = Lv.size() - 2;   // number of detail levels

    auto energy = [&](size_t off, size_t len) {
        double s = 0.0;
        for (size_t i = off; i < off + len && i < Cv.size(); ++i) s += Cv[i] * Cv[i];
        return s;
    };

    double total = 0.0;
    for (double c : Cv) total += c * c;
    const double scale = (total > 0.0) ? 100.0 / total : 0.0;

    // Approximation band = first L(0) coefficients.
    const size_t aLen = static_cast<size_t>(Lv[0]);
    Value Ea = Value::scalar(energy(0, aLen) * scale, mr);

    // C packs details coarsest-first (cD_N … cD_1 at L(1) … L(nLev)); MATLAB's
    // Ed is ordered finest-first (level 1 … level N), so store reversed.
    std::vector<double> Ed(nLev, 0.0);
    size_t off = aLen;
    for (size_t k = 0; k < nLev; ++k) {
        const size_t len = static_cast<size_t>(Lv[1 + k]);   // band cD_{N-k} = level (N-k)
        Ed[nLev - 1 - k] = energy(off, len) * scale;
        off += len;
    }

    WenergyResult out;
    out.Ea = std::move(Ea);
    out.Ed = rowFromVec(Ed, mr);
    return out;
}

namespace {
// Full convolution y[n] = Σ_k a[k]·b[n−k], length |a|+|b|−1.
std::vector<double> convFull(const std::vector<double> &a,
                             const std::vector<double> &b) {
    if (a.empty() || b.empty()) return {};
    std::vector<double> c(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            c[i + j] += a[i] * b[j];
    return c;
}
} // anonymous

Value upcoef(const std::string &type, const Value &X, const std::string &wname,
             int n, long long len, std::pmr::memory_resource *mr)
{
    if (type != "a" && type != "d")
        throw Error("upcoef: type must be 'a' (approximation) or 'd' (detail)",
                    0, 0, "upcoef", "", "numkit:upcoef:type");
    if (n < 0)
        throw Error("upcoef: N must be >= 0",
                    0, 0, "upcoef", "", "numkit:upcoef:N");

    auto fb = wavelet_filters(wname);
    auto y = vecFromValue(X);
    const bool detail = (type == "d");

    for (int i = 0; i < n; ++i) {
        // First level of a detail branch uses the highpass; all else lowpass.
        const std::vector<double> &F = (i == 0 && detail) ? fb.Hi_R : fb.Lo_R;
        std::vector<double> up;
        if (!y.empty()) {
            up.assign(2 * y.size() - 1, 0.0);          // [y0, 0, y1, 0, …, y_{n-1}]
            for (size_t k = 0; k < y.size(); ++k) up[2 * k] = y[k];
        }
        y = convFull(up, F);
    }

    // Optional: keep the central `len` samples (MATLAB wkeep convention).
    if (len >= 0 && static_cast<size_t>(len) < y.size()) {
        const size_t start = (y.size() - static_cast<size_t>(len)) / 2;
        y = std::vector<double>(y.begin() + start,
                                y.begin() + start + static_cast<size_t>(len));
    }
    return rowFromVec(y, mr);
}

namespace {

void check_wptree(const Value &tree, const char *fn)
{
    if (!tree.isStruct() || !tree.hasField("nodes") || !tree.hasField("isLeaf") ||
        !tree.hasField("wname") || !tree.hasField("depth")) {
        throw Error(std::string(fn) + ": T must be a wavelet packet tree struct from wpdec",
                    0, 0, fn, "", std::string("numkit:") + fn + ":invalidTree");
    }
}

size_t parse_wp_node(const Value &node, const char *fn)
{
    if (!node.isNumeric() || node.isComplex() || node.isEmpty())
        throw Error(std::string(fn) + ": NODE must be a non-negative integer or [depth, pos] pair",
                    0, 0, fn, "", std::string("numkit:") + fn + ":invalidNode");
    if (node.numel() == 1) {
        const double v = node.elemAsDouble(0);
        if (!std::isfinite(v) || v < 0.0 || std::floor(v) != v)
            throw Error(std::string(fn) + ": NODE index must be a non-negative integer",
                        0, 0, fn, "", std::string("numkit:") + fn + ":invalidNode");
        return static_cast<size_t>(v);
    }
    if (node.numel() == 2) {
        const double d = node.elemAsDouble(0);
        const double p = node.elemAsDouble(1);
        if (!std::isfinite(d) || d < 0.0 || std::floor(d) != d || d > 20.0 ||
            !std::isfinite(p) || p < 0.0 || std::floor(p) != p) {
            throw Error(std::string(fn) + ": [depth, pos] must be non-negative integers",
                        0, 0, fn, "", std::string("numkit:") + fn + ":invalidNode");
        }
        const size_t depth = static_cast<size_t>(d);
        const size_t pos = static_cast<size_t>(p);
        const size_t levelStart = (1ULL << depth) - 1;
        if (pos >= (1ULL << depth))
            throw Error(std::string(fn) + ": position out of range for depth",
                        0, 0, fn, "", std::string("numkit:") + fn + ":invalidNode");
        return levelStart + pos;
    }
    throw Error(std::string(fn) + ": NODE must be a scalar index or 2-element [depth, pos] vector",
                0, 0, fn, "", std::string("numkit:") + fn + ":invalidNode");
}

} // anonymous

Value wpdec(const Value &x, int n, const std::string &wname,
            const std::string &entName, double entPar,
            std::pmr::memory_resource *mr)
{
    if (!x.isNumeric() || x.isComplex() || x.isEmpty() ||
        x.dims().ndims() > 2 || (x.dims().rows() > 1 && x.dims().cols() > 1)) {
        throw Error("wpdec: X must be a non-empty real 1-D vector",
                    0, 0, "wpdec", "", "numkit:wpdec:expectedVector");
    }
    if (n < 1 || n > 16)
        throw Error("wpdec: N must be an integer in [1, 16]",
                    0, 0, "wpdec", "", "numkit:wpdec:invalidLevel");

    // Validate wavelet filters up front.
    (void)wavelet_filters(wname);

    const size_t totalNodes = (1ULL << (n + 1)) - 1;
    const size_t firstLeaf = (1ULL << n) - 1;
    const bool isCol = (x.dims().rows() > 1 && x.dims().cols() == 1);

    Value nodes = Value::cell(1, totalNodes, mr);
    Value isLeaf = Value::matrix(1, totalNodes, ValueType::DOUBLE, mr);
    Value entropy = Value::matrix(1, totalNodes, ValueType::DOUBLE, mr);
    double *leafData = isLeaf.doubleDataMut();
    double *entData = entropy.doubleDataMut();
    std::fill(leafData, leafData + totalNodes, 0.0);
    std::fill(entData, entData + totalNodes, 0.0);

    // Orient node coefficients to match input orientation (row or col).
    auto orientLikeX = [&](const Value &v) -> Value {
        if (isCol && v.dims().rows() == 1 && v.dims().cols() > 1)
            return v.reshape(v.dims().cols(), 1, mr);
        if (!isCol && v.dims().cols() == 1 && v.dims().rows() > 1)
            return v.reshape(1, v.dims().rows(), mr);
        return v;
    };

    nodes.cellAt(0) = orientLikeX(x);
    entData[0] = wentropy(nodes.cellAt(0), entName, entPar, mr).toScalar();

    for (size_t k = 0; k < firstLeaf; ++k) {
        auto [cA, cD] = dwt(nodes.cellAt(k), wname, mr);
        const size_t left = 2 * k + 1;
        const size_t right = 2 * k + 2;
        nodes.cellAt(left) = orientLikeX(cA);
        nodes.cellAt(right) = orientLikeX(cD);
        entData[left] = wentropy(nodes.cellAt(left), entName, entPar, mr).toScalar();
        entData[right] = wentropy(nodes.cellAt(right), entName, entPar, mr).toScalar();
    }
    for (size_t k = firstLeaf; k < totalNodes; ++k)
        leafData[k] = 1.0;

    Value t = Value::structure(mr);
    t.setFieldAll("order", Value::scalar(2.0, mr));
    t.setFieldAll("depth", Value::scalar(static_cast<double>(n), mr));
    t.setFieldAll("wname", Value::fromString(wname, mr));
    t.setFieldAll("entName", Value::fromString(entName, mr));
    t.setFieldAll("entPar", Value::scalar(entPar, mr));
    t.setFieldAll("isColumn", Value::logicalScalar(isCol, mr));
    t.setFieldAll("nodes", std::move(nodes));
    t.setFieldAll("isLeaf", std::move(isLeaf));
    t.setFieldAll("entropy", std::move(entropy));
    return t;
}

Value wpcoef(const Value &tree, const Value &node,
             std::pmr::memory_resource *mr)
{
    check_wptree(tree, "wpcoef");
    const size_t idx = parse_wp_node(node, "wpcoef");
    const Value &nodes = tree.field("nodes");
    if (idx >= nodes.numel() || nodes.cellAt(idx).isEmpty())
        throw Error("wpcoef: node " + std::to_string(idx) + " does not exist in tree",
                    0, 0, "wpcoef", "", "numkit:wpcoef:nonexistentNode");
    (void)mr;
    return nodes.cellAt(idx);
}

Value wprec(const Value &tree, std::pmr::memory_resource *mr)
{
    check_wptree(tree, "wprec");
    const std::string wname = tree.field("wname").toString();
    const bool isCol = tree.hasField("isColumn") && tree.field("isColumn").toBool();
    const Value &nodes = tree.field("nodes");
    const Value &isLeaf = tree.field("isLeaf");
    const size_t numNodes = nodes.numel();

    std::vector<Value> rec(numNodes);
    for (size_t k = 0; k < numNodes; ++k)
        rec[k] = nodes.cellAt(k);

    for (size_t k = numNodes; k-- > 0;) {
        if (rec[k].isEmpty()) continue;
        if (isLeaf.elemAsDouble(k) != 0.0) continue;
        const size_t left = 2 * k + 1;
        const size_t right = 2 * k + 2;
        if (left < numNodes && right < numNodes &&
            !rec[left].isEmpty() && !rec[right].isEmpty()) {
            const long long targetLen = static_cast<long long>(rec[k].numel());
            Value r = idwt(rec[left], rec[right], wname, targetLen, mr);
            if (isCol && r.dims().rows() == 1 && r.dims().cols() > 1)
                r = r.reshape(r.dims().cols(), 1, mr);
            rec[k] = std::move(r);
        }
    }
    return rec[0];
}

WpspltResult wpsplt(const Value &tree, const Value &node,
                    std::pmr::memory_resource *mr)
{
    check_wptree(tree, "wpsplt");
    const size_t idx = parse_wp_node(node, "wpsplt");
    const Value &oldNodes = tree.field("nodes");
    const Value &oldLeaf = tree.field("isLeaf");
    const Value &oldEnt = tree.field("entropy");
    if (idx >= oldNodes.numel() || oldNodes.cellAt(idx).isEmpty())
        throw Error("wpsplt: node " + std::to_string(idx) + " does not exist in tree",
                    0, 0, "wpsplt", "", "numkit:wpsplt:nonexistentNode");
    if (oldLeaf.elemAsDouble(idx) == 0.0)
        throw Error("wpsplt: node " + std::to_string(idx) + " is not a terminal leaf node",
                    0, 0, "wpsplt", "", "numkit:wpsplt:notTerminalNode");

    const std::string wname = tree.field("wname").toString();
    const std::string entName = tree.hasField("entName") ? tree.field("entName").toString() : "shannon";
    const double entPar = tree.hasField("entPar") ? tree.field("entPar").toScalar() : 0.0;
    const bool isCol = tree.hasField("isColumn") && tree.field("isColumn").toBool();

    const size_t left = 2 * idx + 1;
    const size_t right = 2 * idx + 2;
    int depth = static_cast<int>(tree.field("depth").toScalar());
    size_t newTotal = oldNodes.numel();
    while (right >= newTotal) {
        ++depth;
        newTotal = (1ULL << (depth + 1)) - 1;
    }

    Value nodes = Value::cell(1, newTotal, mr);
    Value isLeaf = Value::matrix(1, newTotal, ValueType::DOUBLE, mr);
    Value entropy = Value::matrix(1, newTotal, ValueType::DOUBLE, mr);
    double *leafData = isLeaf.doubleDataMut();
    double *entData = entropy.doubleDataMut();
    std::fill(leafData, leafData + newTotal, 0.0);
    std::fill(entData, entData + newTotal, 0.0);

    for (size_t k = 0; k < oldNodes.numel(); ++k) {
        nodes.cellAt(k) = oldNodes.cellAt(k);
        leafData[k] = oldLeaf.elemAsDouble(k);
        entData[k] = oldEnt.elemAsDouble(k);
    }

    auto [cA, cD] = dwt(nodes.cellAt(idx), wname, mr);
    if (isCol) {
        if (cA.dims().rows() == 1 && cA.dims().cols() > 1) cA = cA.reshape(cA.dims().cols(), 1, mr);
        if (cD.dims().rows() == 1 && cD.dims().cols() > 1) cD = cD.reshape(cD.dims().cols(), 1, mr);
    }
    leafData[idx] = 0.0;
    leafData[left] = 1.0;
    leafData[right] = 1.0;
    nodes.cellAt(left) = cA;
    nodes.cellAt(right) = cD;
    entData[left] = wentropy(cA, entName, entPar, mr).toScalar();
    entData[right] = wentropy(cD, entName, entPar, mr).toScalar();

    Value outTree = Value::structure(mr);
    outTree.setFieldAll("order", Value::scalar(2.0, mr));
    outTree.setFieldAll("depth", Value::scalar(static_cast<double>(depth), mr));
    outTree.setFieldAll("wname", Value::fromString(wname, mr));
    outTree.setFieldAll("entName", Value::fromString(entName, mr));
    outTree.setFieldAll("entPar", Value::scalar(entPar, mr));
    outTree.setFieldAll("isColumn", Value::logicalScalar(isCol, mr));
    outTree.setFieldAll("nodes", std::move(nodes));
    outTree.setFieldAll("isLeaf", std::move(isLeaf));
    outTree.setFieldAll("entropy", std::move(entropy));

    return {std::move(outTree), std::move(cA), std::move(cD)};
}

BesttreeResult besttree(const Value &tree, std::pmr::memory_resource *mr)
{
    check_wptree(tree, "besttree");
    const Value &oldNodes = tree.field("nodes");
    const Value &oldLeaf = tree.field("isLeaf");
    const Value &oldEnt = tree.field("entropy");
    const size_t numNodes = oldNodes.numel();

    std::vector<Value> nodesVec(numNodes);
    std::vector<double> leafVec(numNodes, 0.0);
    std::vector<double> optEnt(numNodes, 0.0);
    for (size_t k = 0; k < numNodes; ++k) {
        nodesVec[k] = oldNodes.cellAt(k);
        leafVec[k] = oldLeaf.elemAsDouble(k);
        optEnt[k] = oldEnt.elemAsDouble(k);
    }

    auto clearSubtree = [&](auto &self, size_t u) -> void {
        const size_t l = 2 * u + 1;
        const size_t r = 2 * u + 2;
        if (l < numNodes && !nodesVec[l].isEmpty()) {
            nodesVec[l] = Value();
            leafVec[l] = 0.0;
            self(self, l);
        }
        if (r < numNodes && !nodesVec[r].isEmpty()) {
            nodesVec[r] = Value();
            leafVec[r] = 0.0;
            self(self, r);
        }
    };

    for (size_t k = numNodes; k-- > 0;) {
        if (nodesVec[k].isEmpty() || leafVec[k] != 0.0) continue;
        const size_t left = 2 * k + 1;
        const size_t right = 2 * k + 2;
        if (left < numNodes && right < numNodes &&
            !nodesVec[left].isEmpty() && !nodesVec[right].isEmpty()) {
            const double childEnt = optEnt[left] + optEnt[right];
            if (childEnt < optEnt[k]) {
                optEnt[k] = childEnt;
            } else {
                // Prune subtree below k.
                leafVec[k] = 1.0;
                clearSubtree(clearSubtree, k);
            }
        }
    }

    Value nodes = Value::cell(1, numNodes, mr);
    Value isLeaf = Value::matrix(1, numNodes, ValueType::DOUBLE, mr);
    Value entropy = Value::matrix(1, numNodes, ValueType::DOUBLE, mr);
    double *leafData = isLeaf.doubleDataMut();
    double *entData = entropy.doubleDataMut();

    std::vector<double> activeEnts;
    for (size_t k = 0; k < numNodes; ++k) {
        nodes.cellAt(k) = nodesVec[k];
        leafData[k] = leafVec[k];
        entData[k] = optEnt[k];
        if (!nodesVec[k].isEmpty())
            activeEnts.push_back(optEnt[k]);
    }

    Value outTree = Value::structure(mr);
    outTree.setFieldAll("order", tree.field("order"));
    outTree.setFieldAll("depth", tree.field("depth"));
    outTree.setFieldAll("wname", tree.field("wname"));
    outTree.setFieldAll("entName", tree.field("entName"));
    outTree.setFieldAll("entPar", tree.field("entPar"));
    outTree.setFieldAll("isColumn", tree.field("isColumn"));
    outTree.setFieldAll("nodes", std::move(nodes));
    outTree.setFieldAll("isLeaf", std::move(isLeaf));
    outTree.setFieldAll("entropy", std::move(entropy));

    Value entCol = Value::matrix(activeEnts.size(), 1, ValueType::DOUBLE, mr);
    if (!activeEnts.empty())
        std::copy(activeEnts.begin(), activeEnts.end(), entCol.doubleDataMut());

    return {std::move(outTree), std::move(entCol)};
}

} // namespace numkit::wavelet
