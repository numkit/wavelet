// toolboxes/wavelet/src/filter/biorfilt.cpp
//
// Biorthogonal (bior*) and reverse-biorthogonal (rbio*) filter bank lookup
// delegating to keel::wavelet::lookup_filters.

#include <keel/ref.h>
#include <keel/wavelet.h>
#include <numkit/wavelet/filter/wfilters.hpp>

#include <utility>

namespace numkit::wavelet {

bool bior_filterbank(keel::StringRef name, FilterBank &out) {
    if (!name.starts_with("bior") && !name.starts_with("rbio")) return false;
    keel::wavelet::FilterBank kfb;
    if (!keel::wavelet::lookup_filters(name, kfb)) return false;
    out.Lo_D = std::move(kfb.lo_d);
    out.Hi_D = std::move(kfb.hi_d);
    out.Lo_R = std::move(kfb.lo_r);
    out.Hi_R = std::move(kfb.hi_r);
    return true;
}

} // namespace numkit::wavelet
