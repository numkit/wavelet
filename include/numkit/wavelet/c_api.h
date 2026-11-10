/* include/numkit/wavelet/c_api.h
 *
 * Self-contained C-ABI for the `numkit/wavelet` toolbox.
 *
 * Supports both:
 *   1. Pre-resolved O(1) dispatch (`numkit_wavelet_resolve` + `numkit_wavelet_invoke`)
 *      with zero string hashing on the hot path.
 *   2. Dynamic name/method dispatch (`numkit_wavelet_call`) for MATLAB overloads,
 *      Name-Value pairs, and toolbox object methods.
 */
#ifndef NUMKIT_WAVELET_C_API_H
#define NUMKIT_WAVELET_C_API_H

#include <numkit/value/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(NUMKIT_PLUGIN_BUILD)
#    define NUMKIT_WAVELET_API __declspec(dllexport)
#  elif defined(NUMKIT_WAVELET_STATIC)
#    define NUMKIT_WAVELET_API
#  else
#    define NUMKIT_WAVELET_API __declspec(dllimport)
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define NUMKIT_WAVELET_API __attribute__((visibility("default")))
#  else
#    define NUMKIT_WAVELET_API
#  endif
#endif

/* Opaque isolated `wavelet` toolbox context */
typedef struct numkit_wavelet_ctx_s numkit_wavelet_ctx;

/* Create an isolated `wavelet` toolbox context (engine-free, zero interpreter overhead). */
NUMKIT_WAVELET_API numkit_wavelet_ctx *numkit_wavelet_ctx_create(numkit_error *err);

/* Destroy `wavelet` toolbox context and release its resources. */
NUMKIT_WAVELET_API void numkit_wavelet_ctx_destroy(numkit_wavelet_ctx *ctx);

/* Seed the `wavelet` context's random number stream (MATLAB `rng(seed)` semantics). */
NUMKIT_WAVELET_API void numkit_wavelet_seed_rng(numkit_wavelet_ctx *ctx, uint64_t seed);

/* Return 1 if function `name` is registered in `wavelet`, 0 otherwise. */
NUMKIT_WAVELET_API int numkit_wavelet_has_function(const numkit_wavelet_ctx *ctx, const char *name);

/* Register a host-language callback (`numkit_host_fn`) into this context. */
NUMKIT_WAVELET_API int numkit_wavelet_register_fn(numkit_wavelet_ctx *ctx,
                                            const char *name,
                                            numkit_host_fn fn,
                                            void *user_data,
                                            numkit_error *err);

/* Resolve function `name` once into an O(1) function pointer (`NULL` if not found). */
NUMKIT_WAVELET_API const numkit_fn *numkit_wavelet_resolve(const numkit_wavelet_ctx *ctx,
                                                     const char *name);

/* Invoke a pre-resolved function pointer directly with zero lookup overhead. */
NUMKIT_WAVELET_API int numkit_wavelet_invoke(numkit_wavelet_ctx *ctx,
                                       const numkit_fn *fn,
                                       const numkit_val *const *args,
                                       size_t nargs,
                                       size_t nargout,
                                       numkit_val **outs,
                                       numkit_error *err);

/* Dynamically invoke `wavelet` function or object method `name`. */
NUMKIT_WAVELET_API int numkit_wavelet_call(numkit_wavelet_ctx *ctx,
                                     const char *name,
                                     const numkit_val *const *args,
                                     size_t nargs,
                                     size_t nargout,
                                     numkit_val **outs,
                                     numkit_error *err);

/* Format a value or `wavelet` class object (e.g. `sym`, `tf`, `ss`) to UTF-8 text. */
NUMKIT_WAVELET_API size_t numkit_wavelet_format(const numkit_wavelet_ctx *ctx,
                                          const numkit_val *v,
                                          char *buf,
                                          size_t buf_cap);

#ifdef __cplusplus
}
#endif

#endif /* NUMKIT_WAVELET_C_API_H */
