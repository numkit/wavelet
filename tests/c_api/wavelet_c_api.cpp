#include <numkit/wavelet/c_api.h>
#include <gtest/gtest.h>

TEST(NumkitWaveletCApi, DbwavfScalingWeights)
{
    const int64_t base = numkit_debug_live_values();
    numkit_error err{};
    numkit_wavelet_ctx *ctx = numkit_wavelet_ctx_create(&err);
    ASSERT_NE(ctx, nullptr) << err.message;

    const numkit_fn *fn = numkit_wavelet_resolve(ctx, "dbwavf");
    ASSERT_NE(fn, nullptr);
    numkit_val *wname = numkit_val_char("db2");
    const numkit_val *args[1] = {wname};
    numkit_val *w = nullptr;
    ASSERT_EQ(numkit_wavelet_invoke(ctx, fn, args, 1, 1, &w, &err), 0) << err.message;
    EXPECT_EQ(numkit_val_numel(w), 4u);

    numkit_val_release(w);
    numkit_val_release(wname);
    numkit_wavelet_ctx_destroy(ctx);
    EXPECT_EQ(numkit_debug_live_values(), base);
}
