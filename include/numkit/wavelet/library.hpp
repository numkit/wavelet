/// @file library.hpp
/// @ingroup group_wavelet
#pragma once

namespace numkit {

/// @addtogroup group_wavelet
class PluginContext;

class WaveletLibrary
{
public:
    static void install(PluginContext &engine);
};

/// @}
} // namespace numkit
