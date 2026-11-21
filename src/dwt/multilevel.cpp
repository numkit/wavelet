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

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/wavelet.h>
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
#include <cstring>
#include <utility>

namespace numkit::wavelet {

namespace {

Value rowFromVec(keel::ArrayRef<double> v,
                 keel::Allocator *mr) {
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

keel::Array<double> vecFromValue(const Value &v, keel::Allocator *mr = nullptr) {
    keel::Array<double> out(v.numel(), mr);
    for (keel::usize i = 0; i < v.numel(); ++i) out[i] = v.elemAsDouble(i);
    return out;
}

void wavedec_argcheck(int n)
{
    if (n < 1)
        throw Error("wavedec: level must be ≥ 1",
                    0, 0, "wavedec", "", "numkit:wavedec:level");
}

void waverec_argcheck(const Value &L)
{
    if (L.numel() < 3)
        throw Error("waverec: L must have at least 3 entries (1 level)",
                    0, 0, "waverec", "", "numkit:waverec:size");
}

void detcoef_argcheck(const Value &L, int level)
{
    const keel::usize Lcount = L.numel();
    if (Lcount < 3)
        throw Error("detcoef: L too short",
                    0, 0, "detcoef", "", "numkit:detcoef:size");
    const int nMax = keel::as<int>(Lcount) - 2;
    if (level < 1 || level > nMax)
        throw Error("detcoef: level out of range",
                    0, 0, "detcoef", "", "numkit:detcoef:level");
}

void wenergy_argcheck(const Value &L)
{
    if (L.numel() < 3)
        throw Error("wenergy: L must have at least 3 entries (a + 1 detail + length)",
                    0, 0, "wenergy", "", "numkit:wenergy:L");
}

void upcoef_argcheck(keel::StringRef type, int n)
{
    if (type != "a" && type != "d")
        throw Error("upcoef: type must be 'a' (approximation) or 'd' (detail)",
                    0, 0, "upcoef", "", "numkit:upcoef:type");
    if (n < 0)
        throw Error("upcoef: N must be >= 0",
                    0, 0, "upcoef", "", "numkit:upcoef:N");
}

void wpdec_argcheck(const Value &x, int n)
{
    if (!x.isNumeric() || x.isComplex() || x.isEmpty() ||
        x.dims().ndims() > 2 || (x.dims().rows() > 1 && x.dims().cols() > 1)) {
        throw Error("wpdec: X must be a non-empty real 1-D vector",
                    0, 0, "wpdec", "", "numkit:wpdec:expectedVector");
    }
    if (n < 1 || n > 16)
        throw Error("wpdec: N must be an integer in [1, 16]",
                    0, 0, "wpdec", "", "numkit:wpdec:invalidLevel");
}

} // anonymous

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

keel::Array<keel::usize> readL(const Value &L, keel::Allocator *mr)
{
    keel::Array<keel::usize> lv(L.numel(), mr);
    for (keel::usize i = 0; i < L.numel(); ++i)
        lv[i] = keel::as<keel::usize>(L.elemAsDouble(i));
    return lv;
}

} // anonymous

std::pair<Value, Value>
wavedec(const Value &x, int n, keel::StringRef wname,
        keel::Allocator *mr)
{
    wavedec_argcheck(n);
    auto fb = lookup_keel_fb(wname, mr);
    auto xv = vecFromValue(x, mr);
    auto r = keel::wavelet::wavedec(xv, keel::as<keel::i32>(n), fb, mr);

    Value L = Value::matrix(1, r.l.size(), ValueType::DOUBLE, mr);
    double *ld = L.doubleDataMut();
    for (keel::usize i = 0; i < r.l.size(); ++i) ld[i] = keel::as<double>(r.l[i]);
    return {rowFromVec(r.c, mr), std::move(L)};
}

Value waverec(const Value &C, const Value &L, keel::StringRef wname,
              keel::Allocator *mr)
{
    waverec_argcheck(L);
    const keel::usize Lcount = L.numel();
    const int n = keel::as<int>(Lcount) - 2;
    auto lv = readL(L, mr);
    auto Cv = vecFromValue(C, mr);

    keel::usize off = lv[0];
    if (off > Cv.size())
        throw Error("waverec: C/L mismatch (approx)",
                    0, 0, "waverec", "", "numkit:waverec:bounds");
    for (int k = 0; k < n; ++k) {
        off += lv[1 + keel::as<keel::usize>(k)];
        if (off > Cv.size())
            throw Error("waverec: C/L mismatch (detail)",
                        0, 0, "waverec", "", "numkit:waverec:bounds");
    }

    auto fb = lookup_keel_fb(wname, mr);
    auto rec = keel::wavelet::waverec(Cv, lv, fb, mr);
    return rowFromVec(rec, mr);
}

static Value appcoef_with_filters(const Value &C, const Value &L,
                                  keel::ArrayRef<double> Lo_R,
                                  keel::ArrayRef<double> Hi_R,
                                  int level,
                                  keel::Allocator *mr)
{
    const keel::usize Lcount = L.numel();
    if (Lcount < 3)
        throw Error("appcoef: L too short",
                    0, 0, "appcoef", "", "numkit:appcoef:size");
    const int nMax = keel::as<int>(Lcount) - 2;
    if (level < 0) level = nMax;
    if (level < 0 || level > nMax)
        throw Error("appcoef: level out of range",
                    0, 0, "appcoef", "", "numkit:appcoef:level");

    auto Cv = vecFromValue(C, mr);
    auto lv = readL(L, mr);
    auto rec = keel::wavelet::appcoef(Cv, lv, Lo_R, Hi_R, keel::as<keel::i32>(level), mr);
    return rowFromVec(rec, mr);
}

Value appcoef(const Value &C, const Value &L, keel::StringRef wname,
              int level,
              keel::Allocator *mr)
{
    auto fb = wavelet_filters(wname);
    return appcoef_with_filters(C, L, fb.Lo_R, fb.Hi_R, level, mr);
}

Value appcoef_with_filters_pub(const Value &C, const Value &L,
                               keel::ArrayRef<double> Lo_R,
                               keel::ArrayRef<double> Hi_R,
                               int level,
                               keel::Allocator *mr)
{
    return appcoef_with_filters(C, L, Lo_R, Hi_R, level, mr);
}

Value detcoef(const Value &C, const Value &L, int level,
              keel::Allocator *mr)
{
    detcoef_argcheck(L, level);
    const keel::usize Lcount = L.numel();
    const int nMax = keel::as<int>(Lcount) - 2;
    auto lv = readL(L, mr);
    auto Cv = vecFromValue(C, mr);

    keel::usize off = lv[0];
    for (int k = 0; k < nMax - level; ++k) off += lv[1 + keel::as<keel::usize>(k)];
    const keel::usize dLen = lv[keel::as<keel::usize>(nMax - level + 1)];
    if (off + dLen > Cv.size())
        throw Error("detcoef: C/L bounds",
                    0, 0, "detcoef", "", "numkit:detcoef:bounds");

    auto d = keel::wavelet::detcoef(Cv, lv, keel::as<keel::i32>(level), mr);
    return rowFromVec(d, mr);
}

WenergyResult wenergy(const Value &C, const Value &L,
                      keel::Allocator *mr)
{
    wenergy_argcheck(L);
    auto Cv = vecFromValue(C, mr);
    auto lv = readL(L, mr);
    const keel::usize nLev = lv.size() - 2;

    double ea = 0.0;
    keel::Array<double> Ed(nLev, 0.0, mr);
    (void)keel::wavelet::wenergy(Cv, lv, ea, Ed);

    WenergyResult out;
    out.Ea = Value::scalar(ea, mr);
    out.Ed = rowFromVec(Ed, mr);
    return out;
}

Value upcoef(keel::StringRef type, const Value &X, keel::StringRef wname,
             int n, long long len, keel::Allocator *mr)
{
    upcoef_argcheck(type, n);
    auto fb = lookup_keel_fb(wname, mr);
    auto xv = vecFromValue(X, mr);
    auto y = keel::wavelet::upcoef(type == "d", xv, fb, keel::as<keel::i32>(n),
                                   keel::as<keel::i64>(len), mr);
    return rowFromVec(y, mr);
}

namespace {

void check_wptree(const Value &tree, const char *fn)
{
    if (!tree.isStruct() || !tree.hasField("nodes") || !tree.hasField("isLeaf") ||
        !tree.hasField("wname") || !tree.hasField("depth")) {
        throw Error(keel::format("{}: T must be a wavelet packet tree struct from wpdec", fn),
                    0, 0, fn, "", keel::format("numkit:{}:invalidTree", fn));
    }
}

keel::usize parse_wp_node(const Value &node, const char *fn)
{
    if (!node.isNumeric() || node.isComplex() || node.isEmpty())
        throw Error(keel::format("{}: NODE must be a non-negative integer or [depth, pos] pair", fn),
                    0, 0, fn, "", keel::format("numkit:{}:invalidNode", fn));
    if (node.numel() == 1) {
        const double v = node.elemAsDouble(0);
        if (!std::isfinite(v) || v < 0.0 || std::floor(v) != v)
            throw Error(keel::format("{}: NODE index must be a non-negative integer", fn),
                        0, 0, fn, "", keel::format("numkit:{}:invalidNode", fn));
        return keel::as<keel::usize>(v);
    }
    if (node.numel() == 2) {
        const double d = node.elemAsDouble(0);
        const double p = node.elemAsDouble(1);
        if (!std::isfinite(d) || d < 0.0 || std::floor(d) != d || d > 20.0 ||
            !std::isfinite(p) || p < 0.0 || std::floor(p) != p) {
            throw Error(keel::format("{}: [depth, pos] must be non-negative integers", fn),
                        0, 0, fn, "", keel::format("numkit:{}:invalidNode", fn));
        }
        const keel::usize depth = keel::as<keel::usize>(d);
        const keel::usize pos = keel::as<keel::usize>(p);
        const keel::usize levelStart = (1ULL << depth) - 1;
        if (pos >= (1ULL << depth))
            throw Error(keel::format("{}: position out of range for depth", fn),
                        0, 0, fn, "", keel::format("numkit:{}:invalidNode", fn));
        return levelStart + pos;
    }
    throw Error(keel::format("{}: NODE must be a scalar index or 2-element [depth, pos] vector", fn),
                0, 0, fn, "", keel::format("numkit:{}:invalidNode", fn));
}

} // anonymous

Value wpdec(const Value &x, int n, keel::StringRef wname,
            keel::StringRef entName, double entPar,
            keel::Allocator *mr)
{
    wpdec_argcheck(x, n);

    // Validate wavelet filters up front.
    (void)wavelet_filters(wname);

    const keel::usize totalNodes = (1ULL << (n + 1)) - 1;
    const keel::usize firstLeaf = (1ULL << n) - 1;
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

    for (keel::usize k = 0; k < firstLeaf; ++k) {
        auto [cA, cD] = dwt(nodes.cellAt(k), wname, mr);
        const keel::usize left = 2 * k + 1;
        const keel::usize right = 2 * k + 2;
        nodes.cellAt(left) = orientLikeX(cA);
        nodes.cellAt(right) = orientLikeX(cD);
        entData[left] = wentropy(nodes.cellAt(left), entName, entPar, mr).toScalar();
        entData[right] = wentropy(nodes.cellAt(right), entName, entPar, mr).toScalar();
    }
    for (keel::usize k = firstLeaf; k < totalNodes; ++k)
        leafData[k] = 1.0;

    Value t = Value::structure(mr);
    t.setFieldAll("order", Value::scalar(2.0, mr));
    t.setFieldAll("depth", Value::scalar(keel::as<double>(n), mr));
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
             keel::Allocator *mr)
{
    check_wptree(tree, "wpcoef");
    const keel::usize idx = parse_wp_node(node, "wpcoef");
    const Value &nodes = tree.field("nodes");
    if (idx >= nodes.numel() || nodes.cellAt(idx).isEmpty())
        throw Error(keel::format("wpcoef: node {} does not exist in tree", idx),
                    0, 0, "wpcoef", "", "numkit:wpcoef:nonexistentNode");
    (void)mr;
    return nodes.cellAt(idx);
}

Value wprec(const Value &tree, keel::Allocator *mr)
{
    check_wptree(tree, "wprec");
    const keel::String wname = tree.field("wname").toString();
    const bool isCol = tree.hasField("isColumn") && tree.field("isColumn").toBool();
    const Value &nodes = tree.field("nodes");
    const Value &isLeaf = tree.field("isLeaf");
    const keel::usize numNodes = nodes.numel();

    keel::Array<Value> rec(numNodes, mr);
    for (keel::usize k = 0; k < numNodes; ++k)
        rec[k] = nodes.cellAt(k);

    for (keel::usize k = numNodes; k-- > 0;) {
        if (rec[k].isEmpty()) continue;
        if (isLeaf.elemAsDouble(k) != 0.0) continue;
        const keel::usize left = 2 * k + 1;
        const keel::usize right = 2 * k + 2;
        if (left < numNodes && right < numNodes &&
            !rec[left].isEmpty() && !rec[right].isEmpty()) {
            const long long targetLen = keel::as<long long>(rec[k].numel());
            Value r = idwt(rec[left], rec[right], wname, targetLen, mr);
            if (isCol && r.dims().rows() == 1 && r.dims().cols() > 1)
                r = r.reshape(r.dims().cols(), 1, mr);
            rec[k] = std::move(r);
        }
    }
    return rec[0];
}

WpspltResult wpsplt(const Value &tree, const Value &node,
                    keel::Allocator *mr)
{
    check_wptree(tree, "wpsplt");
    const keel::usize idx = parse_wp_node(node, "wpsplt");
    const Value &oldNodes = tree.field("nodes");
    const Value &oldLeaf = tree.field("isLeaf");
    const Value &oldEnt = tree.field("entropy");
    if (idx >= oldNodes.numel() || oldNodes.cellAt(idx).isEmpty())
        throw Error(keel::format("wpsplt: node {} does not exist in tree", idx),
                    0, 0, "wpsplt", "", "numkit:wpsplt:nonexistentNode");
    if (oldLeaf.elemAsDouble(idx) == 0.0)
        throw Error(keel::format("wpsplt: node {} is not a terminal leaf node", idx),
                    0, 0, "wpsplt", "", "numkit:wpsplt:notTerminalNode");

    const keel::String wname = tree.field("wname").toString();
    const keel::String entName = tree.hasField("entName") ? tree.field("entName").toString() : keel::String("shannon");
    const double entPar = tree.hasField("entPar") ? tree.field("entPar").toScalar() : 0.0;
    const bool isCol = tree.hasField("isColumn") && tree.field("isColumn").toBool();

    const keel::usize left = 2 * idx + 1;
    const keel::usize right = 2 * idx + 2;
    int depth = keel::as<int>(tree.field("depth").toScalar());
    keel::usize newTotal = oldNodes.numel();
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

    for (keel::usize k = 0; k < oldNodes.numel(); ++k) {
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
    outTree.setFieldAll("depth", Value::scalar(keel::as<double>(depth), mr));
    outTree.setFieldAll("wname", Value::fromString(wname, mr));
    outTree.setFieldAll("entName", Value::fromString(entName, mr));
    outTree.setFieldAll("entPar", Value::scalar(entPar, mr));
    outTree.setFieldAll("isColumn", Value::logicalScalar(isCol, mr));
    outTree.setFieldAll("nodes", std::move(nodes));
    outTree.setFieldAll("isLeaf", std::move(isLeaf));
    outTree.setFieldAll("entropy", std::move(entropy));

    return {std::move(outTree), std::move(cA), std::move(cD)};
}

BesttreeResult besttree(const Value &tree, keel::Allocator *mr)
{
    check_wptree(tree, "besttree");
    const Value &oldNodes = tree.field("nodes");
    const Value &oldLeaf = tree.field("isLeaf");
    const Value &oldEnt = tree.field("entropy");
    const keel::usize numNodes = oldNodes.numel();

    keel::Array<Value> nodesVec(numNodes, mr);
    keel::Array<double> leafVec(numNodes, 0.0, mr);
    keel::Array<double> optEnt(numNodes, 0.0, mr);
    for (keel::usize k = 0; k < numNodes; ++k) {
        nodesVec[k] = oldNodes.cellAt(k);
        leafVec[k] = oldLeaf.elemAsDouble(k);
        optEnt[k] = oldEnt.elemAsDouble(k);
    }

    auto clearSubtree = [&](auto &self, keel::usize u) -> void {
        const keel::usize l = 2 * u + 1;
        const keel::usize r = 2 * u + 2;
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

    for (keel::usize k = numNodes; k-- > 0;) {
        if (nodesVec[k].isEmpty() || leafVec[k] != 0.0) continue;
        const keel::usize left = 2 * k + 1;
        const keel::usize right = 2 * k + 2;
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

    keel::Array<double> activeEnts(mr);
    for (keel::usize k = 0; k < numNodes; ++k) {
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
