// toolboxes/wavelet/src/shape/gauss.cpp
// Real and complex Gaussian wavelets (gauswavf / cgauwavf), delegated to keel::wavelet.

#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/wavelet.h>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/wavelet/shape/shape.hpp>

namespace numkit::wavelet {

namespace {

void gauswavf_argcheck(keel::f64 /*lb*/, keel::f64 /*ub*/, keel::usize /*N*/, int p)
{
    if (p < 1)
        throw Error("gauswavf: derivative order p must be ≥ 1",
                    0, 0, "gauswavf", "", "numkit:gauswavf:p");
}

void cgauwavf_argcheck(keel::f64 /*lb*/, keel::f64 /*ub*/, keel::usize /*N*/, int p)
{
    if (p < 1)
        throw Error("cgauwavf: derivative order p must be ≥ 1",
                    0, 0, "cgauwavf", "", "numkit:cgauwavf:p");
}

} // anonymous

std::tuple<Value, Value>
gauswavf(keel::f64 lb, keel::f64 ub, keel::usize N, int p, keel::Allocator *mr)
{
    gauswavf_argcheck(lb, ub, N, p);
    Value xv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    Value pv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};

    keel::wavelet::gauswavf(lb, ub, keel::as<keel::i32>(p),
                            keel::MutArrayRef<keel::f64>(pv.doubleDataMut(), N),
                            keel::MutArrayRef<keel::f64>(xv.doubleDataMut(), N));
    return {std::move(pv), std::move(xv)};
}

std::tuple<Value, Value>
cgauwavf(keel::f64 lb, keel::f64 ub, keel::usize N, int p, keel::Allocator *mr)
{
    cgauwavf_argcheck(lb, ub, N, p);
    Value xv = Value::matrix(1, N, ValueType::DOUBLE, mr);
    Value pv = Value::matrix(1, N, ValueType::COMPLEX, mr);
    if (N == 0) return {std::move(pv), std::move(xv)};

    keel::wavelet::cgauwavf(lb, ub, keel::as<keel::i32>(p),
                            keel::MutArrayRef<keel::c64>(pv.complexDataMut(), N),
                            keel::MutArrayRef<keel::f64>(xv.doubleDataMut(), N), mr);
    return {std::move(pv), std::move(xv)};
}

} // namespace numkit::wavelet
