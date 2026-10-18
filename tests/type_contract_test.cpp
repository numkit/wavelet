// toolboxes/wavelet/tests/type_contract_test.cpp
//
// Class-contract guards for the wavelet argcheck portion (2026-09-19):
// per-entry accepted classes, complex policy, result classes, and the
// probed structural quirks (wkeep's invalid-LEN passthrough, wextend's
// LOC letter table, idwt2's single contagion + central SIZ crop,
// ihaart's class-from-A, detcoef2's class-preserving gather, iswt's
// integer classes, imodwt's complex support).
// Reference: MATLAB R2025b, build/probe41..53 (≈340 mirrored probes).

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <cmath>
#include <gtest/gtest.h>

class WaveletTypeContract : public ::testing::Test
{
public:
    numkit::StandardEngine engine;
    numkit::Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
    std::string evalClass(const std::string &c) { return eval(c).toString(); }
    bool throws(const std::string &c)
    {
        try { eval(c); } catch (...) { return true; }
        return false;
    }
};

// ── wthresh ───────────────────────────────────────────────────────────
TEST_F(WaveletTypeContract, WthreshFloatOnlyComplexComputed)
{
    eval("v = [3 -1 4 1 5 9 2 6];");
    EXPECT_NEAR(evalScalar("wthresh(v,'s',2)(3)"), 2.0, 1e-12);
    // single in, single out — from X or from T (single contagion), including complex
    EXPECT_EQ(evalClass("class(wthresh(single(v),'s',2))"), "single");
    EXPECT_EQ(evalClass("class(wthresh(v,'s',single(2)))"), "single");
    EXPECT_EQ(evalClass("class(wthresh(single(complex(1:4, -(1:4))),'s',1))"), "single");
    EXPECT_EQ(evalClass("class(wkeep(single(complex(1:4, -(1:4))),2))"), "single");
    EXPECT_EQ(evalClass("class(wextend('1','sym',single(complex(1:4, -(1:4))),1))"), "single");
    // complex soft-threshold keeps the phase (probed R2025b)
    eval("y = wthresh([1+2i 3-1i 2],'s',1);");
    EXPECT_NEAR(eval("real(y(1))").toScalar(), 0.552786, 1e-6);
    EXPECT_NEAR(eval("imag(y(1))").toScalar(), 1.105573, 1e-6);
    // elementwise T of the same size computes
    eval("yt = wthresh([1 5 2],'s',[1 2 1]);");
    EXPECT_NEAR(evalScalar("yt(2)"), 3.0, 1e-12);
    // rejected classes
    EXPECT_TRUE(throws("wthresh(logical([1 0 1]),'s',0.5);"));
    EXPECT_TRUE(throws("wthresh(uint8([10 200]),'h',50);"));
    EXPECT_TRUE(throws("wthresh(int16([-5 20]),'s',10);"));
    EXPECT_TRUE(throws("wthresh('ab','s',1);"));
    EXPECT_TRUE(throws("wthresh(v,'s',complex(2));"));
    EXPECT_TRUE(throws("wthresh(v,'s',[1 2]);"));
    // sorh is case-insensitive, one character
    EXPECT_NEAR(evalScalar("wthresh(v,'S',2)(1)"), 1.0, 1e-12);
    EXPECT_TRUE(throws("wthresh(v,'soft',2);"));
    // empty passes through
    EXPECT_EQ(evalScalar("numel(wthresh([],'s',1))"), 0.0);
}

// ── wrev ──────────────────────────────────────────────────────────────
TEST_F(WaveletTypeContract, WrevPreservesEveryClass)
{
    EXPECT_EQ(evalClass("class(wrev(single([5 6 7])))"), "single");
    EXPECT_EQ(evalClass("class(wrev(logical([1 0 1])))"), "logical");
    EXPECT_EQ(evalClass("class(wrev(uint8([5 6 7])))"), "uint8");
    EXPECT_EQ(evalClass("class(wrev(int16([-3 4])))"), "int16");
    EXPECT_EQ(evalClass("class(wrev('abcd'))"), "char");
    EXPECT_EQ(evalClass("class(wrev(int32([5 -6 7])))"), "int32");
    // per-column flip for matrices, elementwise for rows
    eval("m = [1 2 3; 4 5 6];");
    EXPECT_NEAR(evalScalar("wrev(m)(1,1)"), 4.0, 1e-12);
    eval("yr = wrev([1+2i, 3]);");
    EXPECT_NEAR(evalScalar("real(yr(2))"), 1.0, 1e-12);
    EXPECT_NEAR(evalScalar("imag(yr(2))"), 2.0, 1e-12);
}

// ── meyeraux ──────────────────────────────────────────────────────────
TEST_F(WaveletTypeContract, MeyerauxFloatOnlyNonempty)
{
    EXPECT_NEAR(evalScalar("meyeraux(0.5)"), 0.5, 1e-12);
    EXPECT_EQ(evalClass("class(meyeraux(single([0 0.25 1])))"), "single");
    EXPECT_NEAR(evalScalar("meyeraux(single(0.5))"), 0.5f, 1e-6);
    EXPECT_TRUE(throws("meyeraux(logical([0 1]));"));
    EXPECT_TRUE(throws("meyeraux(uint8([0 128]));"));
    EXPECT_TRUE(throws("meyeraux([0.5+0.5i]);"));
    EXPECT_TRUE(throws("meyeraux('ab');"));
    EXPECT_TRUE(throws("meyeraux([]);"));
}

// ── wkeep ─────────────────────────────────────────────────────────────
TEST_F(WaveletTypeContract, WkeepClassPreservingInvalidLenUnchanged)
{
    eval("v = [3 -1 4 1 5 9 2 6];");
    // classes pass through, complex included
    EXPECT_EQ(evalClass("class(wkeep(single(v), 5))"), "single");
    EXPECT_EQ(evalClass("class(wkeep(logical([1 1 0 1 1 0]), 3))"), "logical");
    EXPECT_EQ(evalClass("class(wkeep(uint8([9 8 7 6]), 2, 'l'))"), "uint8");
    eval("yk = wkeep([1+2i 3-1i 5 2], 2);");
    EXPECT_NEAR(evalScalar("imag(yk(1))"), -1.0, 1e-12);   // centered 2 of 4 → [3-1i 5]
    // invalid LEN (non-integer / negative / too big) returns x UNCHANGED
    EXPECT_EQ(evalScalar("numel(wkeep(v, 2.5))"), 8.0);
    EXPECT_EQ(evalScalar("numel(wkeep(v, -1))"), 8.0);
    EXPECT_EQ(evalScalar("numel(wkeep(v, 9))"), 8.0);
    EXPECT_EQ(evalScalar("wkeep(v, 2.5)(3)"), 4.0);
    EXPECT_EQ(evalClass("class(wkeep(single(v), 9))"), "single");
    // len 0 → empty result; vector + 2-element LEN uses the matching dim
    EXPECT_EQ(evalScalar("numel(wkeep(v, 0))"), 0.0);
    eval("y23 = wkeep(v, [2 3]);");
    EXPECT_NEAR(evalScalar("y23(1)"), 4.0, 1e-12);
    EXPECT_EQ(evalScalar("numel(y23)"), 3.0);
    EXPECT_EQ(evalScalar("numel(wkeep(magic(3), [4 3]))"), 9.0);   // unchanged
    // char LEN rejects; explicit FIRST bounds are enforced
    EXPECT_TRUE(throws("wkeep(v, 'x');"));
    EXPECT_TRUE(throws("wkeep(v, 3, 0);"));
    EXPECT_TRUE(throws("wkeep(v, 3, 7);"));
    EXPECT_NEAR(evalScalar("wkeep(v, 3, 6)(1)"), 9.0, 1e-12);
}

// ── wextend ───────────────────────────────────────────────────────────
TEST_F(WaveletTypeContract, WextendIntClassesAndLocTable)
{
    eval("v = [3 -1 4 1 5 9 2 6];");
    eval("m2 = [1 2 3; 4 5 6; 7 8 9];");
    // integer classes pass through with the class preserved
    EXPECT_EQ(evalClass("class(wextend('1','sym',int16(v),2))"), "int16");
    EXPECT_EQ(evalClass("class(wextend('1','sym',uint16(v),2))"), "uint16");
    EXPECT_EQ(evalClass("class(wextend('1','sp0',uint8([5 6 7]),2))"), "uint8");
    // complex extends through the linear modes
    eval("ye = wextend('1','sym',[1+2i 3 4],1);");
    EXPECT_NEAR(evalScalar("imag(ye(1))"), 2.0, 1e-12);
    // rejected: logical, char, negative/non-integer/complex lf
    EXPECT_TRUE(throws("wextend('1','zpd',logical([1 0 1]),2);"));
    EXPECT_TRUE(throws("wextend('1','sym','abcd',2);"));
    EXPECT_TRUE(throws("wextend('1','sym',v,-1);"));
    EXPECT_TRUE(throws("wextend('1','sym',v,2.9);"));
    EXPECT_TRUE(throws("wextend('1','sym',v,complex(2));"));
    // LOC letters: l/u → start only, r/d → end only, n → unchanged,
    // anything else (numbers included) → both. 'spd' = 'sp1' alias.
    EXPECT_EQ(evalScalar("numel(wextend('1','sym',v,2,'l'))"), 10.0);
    EXPECT_EQ(evalScalar("numel(wextend('1','sym',v,2,'u'))"), 10.0);
    EXPECT_EQ(evalScalar("numel(wextend('1','sym',v,2,'r'))"), 10.0);
    EXPECT_EQ(evalScalar("numel(wextend('1','sym',v,2,'d'))"), 10.0);
    EXPECT_EQ(evalScalar("numel(wextend('1','sym',v,2,'b'))"), 12.0);
    EXPECT_EQ(evalScalar("numel(wextend('1','sym',v,2,'n'))"), 8.0);
    EXPECT_EQ(evalScalar("numel(wextend('1','sym',v,2,'z'))"), 12.0);
    EXPECT_EQ(evalScalar("numel(wextend('1','sym',v,2,123))"), 12.0);
    eval("yspd = wextend('1','spd',v,2);");
    EXPECT_NEAR(evalScalar("yspd(1)"), 11.0, 1e-12);
    // 2-element lf for the 2-D forms; type aliases
    EXPECT_EQ(evalScalar("size(wextend('2','sym',m2,[1 2]),1)"), 5.0);
    EXPECT_EQ(evalScalar("size(wextend('2','sym',m2,[1 2]),2)"), 7.0);
    EXPECT_EQ(evalScalar("numel(wextend('1d','sym',v,2))"), 12.0);
    EXPECT_TRUE(throws("wextend('3','sym',v,2);"));
}

// ── idwt2 ─────────────────────────────────────────────────────────────
TEST_F(WaveletTypeContract, Idwt2ClassContagionAndSizCrop)
{
    eval("[cA,cH,cV,cD] = dwt2([1 2 3; 4 5 6; 7 8 9], 'haar');");
    EXPECT_NEAR(evalScalar("idwt2(cA,cH,cV,cD,'haar')(4)"), 7.0, 1e-12);
    // single anywhere → single result; logical/uint8 compute in double
    EXPECT_EQ(evalClass("class(idwt2(single(cA),single(cH),single(cV),single(cD),'haar'))"), "single");
    EXPECT_EQ(evalClass("class(idwt2(cA,single(cH),cV,cD,'haar'))"), "single");
    EXPECT_EQ(evalClass("class(idwt2(logical(round(cA)),logical(round(cH)),logical(round(cV)),logical(round(cD)),'haar'))"), "double");
    EXPECT_EQ(evalClass("class(idwt2(uint8(round(abs(cA))),uint8(round(abs(cH))),uint8(round(abs(cV))),uint8(round(abs(cD))),'haar'))"), "double");
    // complex cA reconstructs a complex result (imag pipeline linear)
    eval("yc = idwt2(complex(cA,1),cH,cV,cD,'haar');");
    EXPECT_NEAR(evalScalar("imag(yc(1))"), 0.5, 1e-12);
    // SIZ = central crop of the natural reconstruction (probed)
    eval("ysx = idwt2(cA,cH,cV,cD,'haar',[2 3]);");
    EXPECT_EQ(evalScalar("size(ysx,1)"), 2.0);
    EXPECT_EQ(evalScalar("size(ysx,2)"), 3.0);
    EXPECT_NEAR(evalScalar("ysx(1,1)"), 4.0, 1e-12);
    // scalar sx rejects; char sx is ignored (natural size)
    EXPECT_TRUE(throws("idwt2(cA,cH,cV,cD,'haar',2);"));
    EXPECT_EQ(evalScalar("numel(idwt2(cA,cH,cV,cD,'haar','xy'))"), 16.0);
    EXPECT_TRUE(throws("idwt2(cA,cH,cV,cD,'haar',[2 3+0.5i]);"));
    // all-empty → empty; ragged sizes reject
    EXPECT_EQ(evalScalar("numel(idwt2([],[],[],[],'haar'))"), 0.0);
    EXPECT_TRUE(throws("idwt2(cA(1:end-1),cH,cV,cD,'haar');"));
}

// ── ihaart ────────────────────────────────────────────────────────────
TEST_F(WaveletTypeContract, IhaartFloatOnlyClassFromA)
{
    eval("[pA,pD] = haart(1:8, 1);");
    EXPECT_NEAR(evalScalar("ihaart(pA,pD)(3)"), 3.0, 1e-12);
    EXPECT_EQ(evalClass("class(ihaart(single(pA),single(pD)))"), "single");
    // the class follows A alone (double A + single D → double)
    EXPECT_EQ(evalClass("class(ihaart(pA,single(pD)))"), "double");
    // complex A computes a complex result; complex D rejects
    eval("ya = ihaart(complex(pA,1),pD);");
    EXPECT_NEAR(evalScalar("imag(ya(1))"), 0.7071068, 1e-6);
    EXPECT_TRUE(throws("ihaart(pA,complex(pD,1));"));
    // logical / integer / char reject on either side
    EXPECT_TRUE(throws("ihaart(logical(pA),logical(pD));"));
    EXPECT_TRUE(throws("ihaart(uint8(round(abs(pA))),uint8(round(abs(pD))));"));
    EXPECT_TRUE(throws("ihaart(int16(round(pA)),int16(round(pD)));"));
}

// ── detcoef2 ──────────────────────────────────────────────────────────
TEST_F(WaveletTypeContract, Detcoef2FreeGatherClassFromC)
{
    eval("[C,S] = wavedec2(magic(8)/10, 2, 'haar');");
    EXPECT_NEAR(evalScalar("detcoef2('h',C,S,1)(1)"), 0.1, 1e-12);
    // C drives the result class — single C → single, uint16 → uint16,
    // complex → complex; S only needs to be numeric REAL
    EXPECT_EQ(evalClass("class(detcoef2('h',single(C),S,1))"), "single");
    EXPECT_EQ(evalClass("class(detcoef2('h',C,single(S),1))"), "double");
    EXPECT_EQ(evalClass("class(detcoef2('h',uint16(abs(C)),S,1))"), "uint16");
    eval("yCx = detcoef2('h',complex(C,1),S,1);");
    EXPECT_NEAR(evalScalar("imag(yCx(1))"), 1.0, 1e-12);
    EXPECT_EQ(evalClass("class(detcoef2('h',logical(C~=0),S,1))"), "logical");
    EXPECT_TRUE(throws("detcoef2('h',C,'x',1);"));
    EXPECT_TRUE(throws("detcoef2('h',C,S,complex(1));"));
    EXPECT_TRUE(throws("detcoef2('q',C,S,1);"));
}

// ── iswt / imodwt ─────────────────────────────────────────────────────
TEST_F(WaveletTypeContract, IswtIntegerClassesOneRowPassthrough)
{
    eval("sw = swt(sin(1:16), 3, 'db2');");
    eval("y = iswt(sw,'db2');");
    EXPECT_NEAR(evalScalar("y(1)"), 0.841471, 1e-5);
    // class preserved incl. integer classes; logical/complex reject
    EXPECT_EQ(evalClass("class(iswt(single(sw),'db2'))"), "single");
    EXPECT_EQ(evalClass("class(iswt(uint8(abs(sw)),'db2'))"), "uint8");
    EXPECT_EQ(evalClass("class(iswt(int16(round(100*sw)),'db2'))"), "int16");
    EXPECT_TRUE(throws("iswt(logical(sw~=0),'db2');"));
    EXPECT_TRUE(throws("iswt(complex(sw,0.5),'db2');"));
    EXPECT_TRUE(throws("iswt([],'db2');"));
    // one-row SWC is an identity passthrough; degenerate lengths reject
    EXPECT_EQ(evalScalar("numel(iswt(zeros(1,8),'db2'))"), 8.0);
    EXPECT_TRUE(throws("iswt(zeros(2,1),'haar');"));
    EXPECT_TRUE(throws("iswt(sin(1:16)','db2');"));   // 16x1: N=1 too short
}

TEST_F(WaveletTypeContract, ImodwtFloatComplexSupport)
{
    eval("w = modwt(sin(1:16), 'db2', 3);");
    eval("y = imodwt(w,'db2');");
    EXPECT_NEAR(evalScalar("y(1)"), 0.841471, 1e-5);
    EXPECT_EQ(evalClass("class(imodwt(single(w),'db2'))"), "single");
    // complex W computes a complex result: the imaginary offset rides
    // the kernel unchanged (imodwt(ones) == 1 — probed both engines).
    eval("yc = imodwt(complex(w,1),'db2');");
    EXPECT_NEAR(evalScalar("imag(yc(1))"), 1.0, 1e-9);
    // integer classes and logical reject
    EXPECT_TRUE(throws("imodwt(int16(round(50*w)),'db2');"));
    EXPECT_TRUE(throws("imodwt(uint8(round(50*w)),'db2');"));
    EXPECT_TRUE(throws("imodwt(logical(w~=0),'db2');"));
    // H ≥ 2 and N ≥ 2 required
    EXPECT_TRUE(throws("imodwt(zeros(1,8),'db2');"));
    EXPECT_TRUE(throws("imodwt(zeros(2,1),'db2');"));
    // a transposed-but-valid matrix computes (no extra shape check)
    EXPECT_EQ(evalScalar("numel(imodwt(w','db2'))"), 4.0);
}
