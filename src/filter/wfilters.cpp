// toolboxes/wavelet/src/filter/wfilters.cpp
//
// Wavelet filter-bank lookup and wfilters() Value adapter delegating to
// keel::wavelet::lookup_filters.

#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/filter/wfilters.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <utility>

namespace numkit::wavelet {

namespace {

Value vecToRow(keel::ArrayRef<keel::f64> v, keel::Allocator *mr) {
    Value r = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), r.doubleDataMut());
    return r;
}

void wfilters_argcheck(keel::StringRef kind)
{
    if (!kind.empty() && kind != "all" && kind != "d" && kind != "r" && kind != "l" && kind != "h")
        throw Error(keel::format("wfilters: unknown kind '{}' "
                                 "(expected 'd', 'r', 'l' or 'h')", kind),
                    0, 0, "wfilters", "", "numkit:wfilters:kind");
}

} // anonymous

FilterBank wavelet_filters(keel::StringRef name) {
    keel::wavelet::FilterBank kfb;
    if (!keel::wavelet::lookup_filters(name, kfb)) {
        throw Error(keel::format("wfilters: unsupported wavelet name '{}' "
                                 "(try haar, db1..db10, sym2..sym10, coif1..coif5, "
                                 "bior1.1..bior6.8, rbio1.1..rbio6.8)", name),
                    0, 0, "wfilters", "", "numkit:wfilters:name");
    }
    FilterBank fb;
    fb.Lo_D = std::move(kfb.lo_d);
    fb.Hi_D = std::move(kfb.hi_d);
    fb.Lo_R = std::move(kfb.lo_r);
    fb.Hi_R = std::move(kfb.hi_r);
    return fb;
}

WFiltersResult wfilters(keel::StringRef name, keel::StringRef kind,
                        keel::Allocator *mr)
{
    wfilters_argcheck(kind);
    auto fb = wavelet_filters(name);
    WFiltersResult r;
    if (kind.empty() || kind == "all") {
        r.Lo_D = vecToRow(fb.Lo_D, mr);
        r.Hi_D = vecToRow(fb.Hi_D, mr);
        r.Lo_R = vecToRow(fb.Lo_R, mr);
        r.Hi_R = vecToRow(fb.Hi_R, mr);
    } else if (kind == "d") {
        r.Lo_D = vecToRow(fb.Lo_D, mr);
        r.Hi_D = vecToRow(fb.Hi_D, mr);
    } else if (kind == "r") {
        r.Lo_R = vecToRow(fb.Lo_R, mr);
        r.Hi_R = vecToRow(fb.Hi_R, mr);
    } else if (kind == "l") {
        r.Lo_D = vecToRow(fb.Lo_D, mr);
        r.Lo_R = vecToRow(fb.Lo_R, mr);
    } else { // "h"
        r.Hi_D = vecToRow(fb.Hi_D, mr);
        r.Hi_R = vecToRow(fb.Hi_R, mr);
    }
    return r;
}

} // namespace numkit::wavelet
