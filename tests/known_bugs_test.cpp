// toolboxes/wavelet/tests/known_bugs_test.cpp
//
// One DISABLED_ test per OPEN bug in bugs/wavelet/*.md. Disabled until
// fixed; remove `DISABLED_` to turn into a live regression guard.
// MATLAB R2025b reference values.

#include <numkit/core/standard_engine.hpp>
#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class WaveletKnownBug : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override {}
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

// bugs/wavelet/wentropy.md — wentropy (Shannon).
// FIXED 2026-06-19 (closed-form entropy) — promoted live.
TEST_F(WaveletKnownBug, WentropyShannon)
{
    EXPECT_NEAR(evalScalar("wentropy([1 2 3 4], 'shannon')"), -69.681618, 1e-4);
}

// bugs/wavelet/ddencmp.md — ddencmp default denoise params.
// FIXED 2026-06-19 (MAD noise estimate + universal threshold) — promoted live.
TEST_F(WaveletKnownBug, Ddencmp)
{
    eval("[thr, sorh, keepapp] = ddencmp('den', 'wv', [1 2 3 8 3 2 1 2]);");
    EXPECT_NEAR(evalScalar("thr"), 2.137920, 1e-5);
    EXPECT_DOUBLE_EQ(evalScalar("keepapp"), 1.0);
}

// bugs/wavelet/dwt-biorthogonal.md — bior2.2 analysis coefficients (FIXED).
TEST_F(WaveletKnownBug, DwtBiorthogonal)
{
    eval("[a, d] = dwt([1 2 3 4 5 6 7 8], 'bior2.2');");
    EXPECT_NEAR(evalScalar("a(1)"), 2.651650429449553, 1e-9);
    EXPECT_NEAR(evalScalar("a(2)"), 1.237436867076458, 1e-9);
    // distinct analysis vs synthesis filters: the lowpass synthesis peak.
    eval("[LoD, HiD, LoR, HiR] = wfilters('bior2.2');");
    EXPECT_NEAR(evalScalar("LoD(4)"),  1.0606601717798214, 1e-12);
    EXPECT_NEAR(evalScalar("LoR(3)"),  0.7071067811865476, 1e-12);
}

// bugs/wavelet/wpdec.md — wavelet packet decomposition exists.
// (Needs a tree type; verify node coefficients vs MATLAB when enabling.)
TEST_F(WaveletKnownBug, DISABLED_WpdecExists)
{
    EXPECT_NO_THROW(eval("t = wpdec([1 2 3 4 5 6 7 8], 2, 'db1');"));
}

// bugs/wavelet/wenergy.md — energy distribution of a decomposition.
// FIXED 2026-06-19 (band energy percentages) — promoted live.
TEST_F(WaveletKnownBug, Wenergy)
{
    eval("[c, l] = wavedec([1 2 3 4 5 6 7 8], 2, 'db1'); [Ea, Ed] = wenergy(c, l);");
    EXPECT_NEAR(evalScalar("Ea"), 95.0980392157, 1e-6);
    EXPECT_NEAR(evalScalar("Ea + sum(Ed)"), 100.0, 1e-6);   // percentages sum to 100
}

// bugs/wavelet/upcoef.md — single-branch coefficient reconstruction.
// FIXED 2026-06-19 (synthesis cascade) — promoted live.
TEST_F(WaveletKnownBug, Upcoef)
{
    eval("y = upcoef('a', 5, 'db1', 2);");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(y)")), 4);
    EXPECT_NEAR(evalScalar("y(1)"), 2.5, 1e-9);
}

// bugs/wavelet/cwt.md — continuous wavelet transform (complex coeff matrix).
TEST_F(WaveletKnownBug, DISABLED_Cwt)
{
    eval("cfs = cwt([1 2 3 4 5 6 7 8 7 6 5 4 3 2 1 0]);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(cfs,2)")), 16);  // time dim == N
    EXPECT_GT(evalScalar("size(cfs,1)"), 1.0);                   // multiple scales
    EXPECT_TRUE(eval("~isreal(cfs)").toBool());                  // complex (Morse)
}

// bugs/wavelet/wavedec2-family.md — 2-D multilevel decomposition + extractors (FIXED).
TEST_F(WaveletKnownBug, Wavedec2Family)
{
    eval("[c, s] = wavedec2(reshape(1:16,4,4), 1, 'db1');");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(c)")), 16);
    EXPECT_NEAR(evalScalar("c(1)"), 7.0, 1e-9);                 // approx coeff
    EXPECT_EQ(static_cast<int>(evalScalar("s(1,1)")), 2);       // size(cA_1)
    eval("H = detcoef2('h', c, s, 1);");
    EXPECT_NEAR(evalScalar("H(1,1)"), -1.0, 1e-9);
    eval("V = detcoef2('v', c, s, 1);");
    EXPECT_NEAR(evalScalar("V(1,1)"), -4.0, 1e-9);
    eval("A = appcoef2(c, s, 'db1', 1);");
    EXPECT_NEAR(evalScalar("A(1,1)"), 7.0, 1e-9);
    EXPECT_NEAR(evalScalar("A(2,2)"), 27.0, 1e-9);
}

// bugs/closed/wavelet/centfrq-scal2frq.md — wavelet center frequency + scale mapping.
TEST_F(WaveletKnownBug, CentfrqScal2frq)
{
    EXPECT_NEAR(evalScalar("centfrq('db4')"),        0.714285714285714, 1e-12); // 5/7
    EXPECT_NEAR(evalScalar("centfrq('db2')"),        0.666666666666667, 1e-12); // 2/3
    EXPECT_NEAR(evalScalar("centfrq('coif2')"),      0.727272727272727, 1e-12); // 8/11
    EXPECT_NEAR(evalScalar("centfrq('sym4')"),       0.714285714285714, 1e-12); // 5/7
    EXPECT_NEAR(evalScalar("centfrq('haar')"),       256.0 / 257.0,     1e-12);
    EXPECT_NEAR(evalScalar("centfrq('haar', 5)"),    32.0 / 33.0,       1e-12);
    EXPECT_NEAR(evalScalar("centfrq('bior2.2')"),    1280.0 / 1279.0,   1e-12);
    EXPECT_NEAR(evalScalar("centfrq('rbio2.2', 5)"), 96.0 / 159.0,      1e-12);
    EXPECT_NEAR(evalScalar("centfrq('morl')"),       0.8125,            1e-12);
    EXPECT_NEAR(evalScalar("centfrq('mexh')"),       0.25,              1e-12);
    EXPECT_NEAR(evalScalar("centfrq('gaus1')"),      0.2,               1e-12);
    EXPECT_NEAR(evalScalar("centfrq('cgau1')"),      0.3,               1e-12);
    EXPECT_NEAR(evalScalar("centfrq('shan1-1.5', 5)"), 0.2,             1e-12);
    EXPECT_NEAR(evalScalar("centfrq('cmor1-1.5', 5)"), 0.4375,          1e-12);
    eval("[f5, x5, r5] = centfrq('db4', 5, 'plot');");
    EXPECT_NEAR(evalScalar("f5"), 5.0 / 7.0, 1e-12);
    EXPECT_EQ(static_cast<int>(evalScalar("numel(x5)")), 225);
    EXPECT_NEAR(evalScalar("r5(1)"), -0.25106497, 1e-6);
    EXPECT_NEAR(evalScalar("scal2frq(4, 'db4', 1)"), 0.178571428571429, 1e-12);
    eval("sf = scal2frq([2; 4; 8], 'db4', 0.5);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(sf, 1)")), 3);
    EXPECT_EQ(static_cast<int>(evalScalar("size(sf, 2)")), 1);
    EXPECT_NEAR(evalScalar("sf(1)"), 5.0 / 7.0, 1e-12);
    EXPECT_TRUE(eval("isa(scal2frq(single([2 4]), 'db4'), 'single')").toBool());
}


// bugs/opened/wavelet/idwt-short-filter-input.md — idwt/idwt2 return an
// empty result when the filter is longer than the coefficients support;
// the reference zero-pads and reconstructs.
TEST_F(WaveletKnownBug, Idwt2Db4ShortCoefficients)
{
    eval("[cA,cH,cV,cD] = dwt2([1 2 3; 4 5 6; 7 8 9], 'haar');");
    eval("y = idwt2(cA, cH, cV, cD, 'db4');");
    // Reference ground truth re-probed 2026-09-18: the natural
    // reconstruction is 10x10 (100 elements — 2·la+Lf-2 per dimension,
    // the wkeep1 uncropped rule), not the 4x4 the bug file first
    // claimed. First row: [0.32821 1.01841 1.36999 1.42223 ...].
    EXPECT_EQ(static_cast<int>(evalScalar("numel(y)")), 100);
    EXPECT_EQ(static_cast<int>(evalScalar("size(y,1)")), 10);
    EXPECT_NEAR(evalScalar("y(1)"), 0.328209, 1e-5);
    EXPECT_NEAR(evalScalar("y(2)"), 1.01841, 1e-5);
    EXPECT_NEAR(evalScalar("y(12)"), 3.1601, 1e-4);
}

// bugs/opened/wavelet/iswt-integer-arithmetic.md — iswt on integer SWC
// computes in double + saturate-casts; the reference runs true integer
// wavelet arithmetic (values diverge).
TEST_F(WaveletKnownBug, IswtUint8HaarValues)
{
    eval("swq = uint8([repmat([100 50 200 25],1,16); "
         "repmat([10 20 30 40],1,16); repmat([60 70 80 90],1,16)]);");
    eval("y = iswt(swq, 'haar');");
    EXPECT_DOUBLE_EQ(evalScalar("y(1)"), 64.0);
    EXPECT_DOUBLE_EQ(evalScalar("y(2)"), 14.0);
    EXPECT_DOUBLE_EQ(evalScalar("y(3)"), 91.0);
    EXPECT_DOUBLE_EQ(evalScalar("y(4)"), 0.0);
}

// bugs/closed/wavelet/iswt-custom-filters.md — custom reconstruction/analysis filters
TEST_F(WaveletKnownBug, IswtCustomFilters)
{
    eval("[LoD, HiD, LoR, HiR] = wfilters('db2');");
    eval("swc = swt(1:16, 2, LoD, HiD);");
    eval("y = iswt(swc, LoR, HiR);");
    EXPECT_DOUBLE_EQ(evalScalar("max(abs(y - iswt(swc, 'db2')))"), 0.0);
    EXPECT_NEAR(evalScalar("max(abs(y - (1:16)))"), 0.0, 1e-10);

    eval("[swa, swd] = swt(1:16, 2, LoD, HiD);");
    eval("y2 = iswt(swa, swd, LoR, HiR);");
    EXPECT_DOUBLE_EQ(evalScalar("max(abs(y2 - y))"), 0.0);
    EXPECT_NEAR(evalScalar("max(abs(y2 - (1:16)))"), 0.0, 1e-10);
}

