// modules/wavelet/tests/arity_bounds_test.cpp
//
// Max-arity rejection guards for the wavelet domain (argcheck migration).
// Every bounded contract gets a max+1 call that must fail with
// numkit:<fn>:maxnargin.

#include <numkit/core/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/value/error.hpp>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

struct Case
{
    const char *fn;
    int maxArgs;
};

const std::vector<Case> kBounds = {
    {"appcoef", 7},
    {"appcoef2", 4},
    {"cgauwavf", 4},
    {"cmorwavf", 5},
    {"coifwavf", 1},
    {"dbwavf", 1},
    {"ddencmp", 3},
    {"detcoef", 4},
    {"detcoef2", 4},
    {"dwt", 5},
    {"dwt2", 2},
    {"dyaddown", 3},
    {"dyadup", 3},
    {"fbspwavf", 6},
    {"gauswavf", 4},
    {"haart", 3},
    {"idwt", 7},
    {"idwt2", 6},
    {"ihaart", 4},
    {"imodwt", 2},
    {"iswt", 4},
    {"mexihat", 3},
    {"meyeraux", 1},
    {"modwt", 3},
    {"morlet", 3},
    {"orthfilt", 1},
    {"qmf", 2},
    {"shanwavf", 5},
    {"swt", 4},
    {"symwavf", 1},
    {"upcoef", 5},
    {"wavedec", 3},
    {"wavedec2", 3},
    {"waverec", 3},
    {"waverec2", 3},
    {"wdenoise", 3},
    {"wenergy", 2},
    {"wentropy", 3},
    {"wextend", 5},
    {"wfilters", 2},
    {"wkeep", 3},
    {"wmaxlev", 2},
    {"wnoisest", 3},
    {"wrcoef", 5},
    {"wrev", 1},
    {"wthresh", 3},
};

std::string callExpr(const Case &c)
{
    std::string s = c.fn;
    s += '(';
    for (int i = 0; i <= c.maxArgs; ++i) {
        if (i)
            s += ", ";
        s += '1';
    }
    s += ')';
    return s;
}

} // namespace

class WaveletArityBoundsTest : public ::testing::TestWithParam<Case>
{
public:
    numkit::StandardEngine engine;
};

TEST_P(WaveletArityBoundsTest, RejectsMaxPlusOne)
{
    const Case &c = GetParam();
    try {
        engine.eval(callExpr(c));
        FAIL() << "expected maxnargin error: " << callExpr(c);
    } catch (const numkit::Error &e) {
        EXPECT_EQ(e.identifier(), std::string("numkit:") + c.fn + ":maxnargin")
            << callExpr(c) << "\n  got: " << e.what();
    }
}

INSTANTIATE_TEST_SUITE_P(WaveletArityBounds, WaveletArityBoundsTest,
                         ::testing::ValuesIn(kBounds));
