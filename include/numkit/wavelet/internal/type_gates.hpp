/// @file type_gates.hpp
/// @ingroup group_wavelet
// toolboxes/wavelet/include/nk/wavelet/internal/type_gates.hpp
//
// Shared <fn>_argcheck cores for toolboxes/wavelet (argcheck separation,
// dev-docs/todo/argcheck_separation.md — boundary rule: raw arguments in,
// all message digestion here; call sites are one pass-through line).
// PRIVATE to wavelet TUs — not part of the public library API.
//
// Rejection classes reference-probed against R2025b on 2026-09-18
// (build/probe41..53, ~340 mirrored class probes). Like image, the
// wavelet toolbox takes a DIFFERENT class set per entry — wthresh's
// signal is double/single only (complex allowed), wkeep/wrev accept
// every array class including char, wextend adds the integer types but
// rejects logical, iswt keeps the integer-wavelet classes while imodwt
// rejects them — so the gates are parameterized by a per-entry mask.
// Complex policy is per entry too (probed): wthresh X / wextend x /
// idwt2 coefficients / detcoef2 C / imodwt compute complex results,
// while meyeraux / ihaart D / iswt are expectedReal.

#pragma once

#include <cmath>
#include <string>
#include <string_view>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (got-clauses)

namespace numkit::wavelet::internal {

/// @addtogroup group_wavelet
/// @{

/// Render "2x3x4" for got-clauses.
inline std::string shapeOf(const Value &v)
{
    const Dims &d = v.dims();
    std::string s = std::to_string(d.dim(0));
    for (int k = 1; k < d.ndim(); ++k) {
        s += "x";
        s += std::to_string(d.dim(k));
    }
    return s;
}

/// Per-entry accepted-class masks, each named after its probe family.
namespace WvClasses {
inline constexpr unsigned FloatOnly   = 1u << 0;   ///< meyeraux, ihaart a/d (double, single)
inline constexpr unsigned IntWavelet  = 1u << 1;   ///< iswt (double, single, all ints; logical rejected)
inline constexpr unsigned AnyNumeric  = 1u << 2;   ///< detcoef2 S, idwt2 coefficients (incl. logical)
inline constexpr unsigned AnyWithChar = 1u << 3;   ///< wrev, wkeep x (any numeric + char passthrough)
}   // namespace WvClasses

/// True when v's base class (complexity aside) is inside mask.
inline bool classAllowed(const Value &v, unsigned mask)
{
    const ValueType t = v.type();
    if (t == ValueType::EMPTY)
        // [] behaves as a 0x0 double — accepted wherever double is;
        // entries that need data voice it via requireNonempty.
        return true;
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX)
        // every family takes (double or complex-)double
        return true;
    if (t == ValueType::SINGLE)
        return (mask & (WvClasses::FloatOnly | WvClasses::IntWavelet
                        | WvClasses::AnyNumeric | WvClasses::AnyWithChar)) != 0;
    if (t == ValueType::LOGICAL)
        return (mask & (WvClasses::AnyNumeric | WvClasses::AnyWithChar)) != 0;
    if (t == ValueType::CHAR)
        return (mask & WvClasses::AnyWithChar) != 0;
    if (isIntegerType(t))
        return (mask & (WvClasses::IntWavelet | WvClasses::AnyNumeric
                        | WvClasses::AnyWithChar)) != 0;
    return false;
}

/// Human listing of a mask's classes (message text).
inline std::string describeClasses(unsigned mask)
{
    if (mask == WvClasses::FloatOnly)
        return "double or single";
    if (mask == WvClasses::IntWavelet)
        return "double, single or an integer class (logical rejected)";
    if (mask == WvClasses::AnyNumeric)
        return "numeric";
    return "an array (numeric or char)";
}

/// @brief The workhorse gate: v must be an array of one of the entry's
/// accepted classes. Wrong class → :inputType; complex → :expectedReal
/// unless allowComplex (the entry's reference computes complex values —
/// the caller handles the complex path or voices its own stub).
inline void requireWaveClass(const Value &v, std::string_view fn,
                             std::string_view argName, int argNum,
                             unsigned mask, bool allowComplex = false)
{
    const ValueType t = v.type();
    if (t == ValueType::COMPLEX && !allowComplex)
        throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                        + " (" + std::string(argName) + ") must be real — "
                        + "complex data is rejected here (got a "
                        + shapeOf(v) + " complex array).",
                    0, 0, std::string(fn), "",
                    "numkit:" + std::string(fn) + ":expectedReal");
    if (!classAllowed(v, mask))
        throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                        + " (" + std::string(argName) + ") must be "
                        + describeClasses(mask) + " (got a " + shapeOf(v)
                        + " " + mtypeName(t) + " array).",
                    0, 0, std::string(fn), "",
                    "numkit:" + std::string(fn) + ":inputType");
}

/// @brief Nonempty gate (meyeraux, iswt): the reference rejects empty
/// arrays with expectedNonempty.
inline void requireNonempty(const Value &v, std::string_view fn,
                            std::string_view argName, int argNum)
{
    if (v.numel() > 0)
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") must be nonempty.",
                0, 0, std::string(fn), "",
                "numkit:" + std::string(fn) + ":expectedNonempty");
}

/// @brief Store one element into a typed matrix (saturation mirrors the
/// reference's integer wavelet arithmetic; values gathered or extrapolated
/// from the same class round-trip exactly).
inline void storeElem(Value &out, std::size_t i, double x)
{
    switch (out.type()) {
        case ValueType::SINGLE:   out.singleDataMut()[i] = static_cast<float>(x); break;
        case ValueType::LOGICAL:  out.logicalDataMut()[i] = x != 0.0 ? 1 : 0; break;
        case ValueType::CHAR:     out.charDataMut()[i] = static_cast<char>(static_cast<int>(x)); break;
        case ValueType::INT8: {
            const double r = std::round(x);
            const double c = r < -128.0 ? -128.0 : (r > 127.0 ? 127.0 : r);
            out.int8DataMut()[i] = static_cast<int8_t>(c);
            break;
        }
        case ValueType::INT16: {
            const double r = std::round(x);
            const double c = r < -32768.0 ? -32768.0 : (r > 32767.0 ? 32767.0 : r);
            out.int16DataMut()[i] = static_cast<int16_t>(c);
            break;
        }
        case ValueType::INT32: {
            const double r = std::round(x);
            const double c = r < -2147483648.0 ? -2147483648.0
                              : (r > 2147483647.0 ? 2147483647.0 : r);
            out.int32DataMut()[i] = static_cast<int32_t>(c);
            break;
        }
        case ValueType::INT64: {
            const double r = std::round(x);
            const double c = r < -9.2233720368547758e18 ? -9.2233720368547758e18
                              : (r > 9.2233720368547758e18 ? 9.2233720368547758e18 : r);
            out.int64DataMut()[i] = static_cast<int64_t>(c);
            break;
        }
        case ValueType::UINT8: {
            const double r = std::round(x);
            const double c = r < 0.0 ? 0.0 : (r > 255.0 ? 255.0 : r);
            out.uint8DataMut()[i] = static_cast<uint8_t>(c);
            break;
        }
        case ValueType::UINT16: {
            const double r = std::round(x);
            const double c = r < 0.0 ? 0.0 : (r > 65535.0 ? 65535.0 : r);
            out.uint16DataMut()[i] = static_cast<uint16_t>(c);
            break;
        }
        case ValueType::UINT32: {
            const double r = std::round(x);
            const double c = r < 0.0 ? 0.0 : (r > 4294967295.0 ? 4294967295.0 : r);
            out.uint32DataMut()[i] = static_cast<uint32_t>(c);
            break;
        }
        case ValueType::UINT64: {
            const double r = std::round(x);
            const double c = r < 0.0 ? 0.0 : (r > 1.8446744073709552e19 ? 1.8446744073709552e19 : r);
            out.uint64DataMut()[i] = static_cast<uint64_t>(c);
            break;
        }
        default:
            out.doubleDataMut()[i] = x;
            break;
    }
}

/// @brief Allocate an output shaped like `like`, typed `target`, with
/// the same page structure (2-D or 3-D only — the wavelet kernels are
/// 1-D/2-D; 3-D inputs keep their pages through wrev/wkeep/meyeraux).
inline Value allocLike(const Value &like, ValueType target,
                       std::pmr::memory_resource *mr = nullptr)
{
    const Dims &d = like.dims();
    return d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), target, mr)
        : Value::matrix(d.rows(), d.cols(), target, mr);
}

/// @brief Copy `src` into a fresh array of the SAME class and shape —
/// the class-preserving "return x unchanged" of the probed wkeep /
/// wextend semantics (invalid LEN returns x, loc 'n' returns x).
inline Value copyInClass(const Value &src,
                         std::pmr::memory_resource *mr = nullptr)
{
    if (src.type() == ValueType::COMPLEX) {
        Value out = Value::complexMatrix(src.dims().rows(), src.dims().cols(), mr);
        const Complex *sd = src.complexData();
        Complex *od = out.complexDataMut();
        for (std::size_t i = 0; i < src.numel(); ++i) od[i] = sd[i];
        return out;
    }
    Value out = allocLike(src, src.type(), mr);
    for (std::size_t i = 0; i < src.numel(); ++i)
        storeElem(out, i, src.elemAsDouble(i));
    return out;
}

/// @brief Gather element i of `src` as double (real part for complex).
inline double gatherReal(const Value &src, std::size_t i)
{
    return src.elemAsDouble(i);
}

/// @} group_wavelet

} // namespace numkit::wavelet::internal
