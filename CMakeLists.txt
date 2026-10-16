cmake_minimum_required(VERSION 3.20)
project(numkit_wavelet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMKIT_WITH_SIMD "Enable Highway SIMD acceleration" ON)

include(FetchContent)

# Dependency: numkit/value
if(NOT TARGET numkit_value)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../value/CMakeLists.txt")
        message(STATUS "Using local numkit/value from ${CMAKE_CURRENT_SOURCE_DIR}/../value")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../value" _deps/numkit_value)
    else()
        FetchContent_Declare(
            numkit_value
            GIT_REPOSITORY https://github.com/numkit/value.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_value)
    endif()
endif()

# Dependency: numkit/sys
if(NOT TARGET numkit_sys)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../sys/CMakeLists.txt")
        message(STATUS "Using local numkit/sys from ${CMAKE_CURRENT_SOURCE_DIR}/../sys")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../sys" _deps/numkit_sys)
    else()
        FetchContent_Declare(
            numkit_sys
            GIT_REPOSITORY https://github.com/numkit/sys.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_sys)
    endif()
endif()

# Dependency: numkit/ops
if(NOT TARGET numkit_ops)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ops/CMakeLists.txt")
        message(STATUS "Using local numkit/ops from ${CMAKE_CURRENT_SOURCE_DIR}/../ops")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ops" _deps/numkit_ops)
    else()
        FetchContent_Declare(
            numkit_ops
            GIT_REPOSITORY https://github.com/numkit/ops.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_ops)
    endif()
endif()

file(GLOB_RECURSE WAVELET_COMPUTE_SOURCES
    "src/denoise/*.cpp"
    "src/dwt/*.cpp"
    "src/filter/*.cpp"
    "src/shape/*.cpp"
    "src/swt/*.cpp"
)

file(GLOB_RECURSE WAVELET_PLUGIN_SOURCES
    "src/register/*.cpp"
    "src/module.cpp"
)

if(NOT NUMKIT_WITH_SIMD)
    list(FILTER WAVELET_COMPUTE_SOURCES EXCLUDE REGEX ".*_highway\\.cpp$")
else()
    list(FILTER WAVELET_COMPUTE_SOURCES EXCLUDE REGEX ".*_portable\\.cpp$")
endif()

add_library(numkit_wavelet_core STATIC ${WAVELET_COMPUTE_SOURCES})
add_library(numkit::wavelet_core ALIAS numkit_wavelet_core)
set_target_properties(numkit_wavelet_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(numkit_wavelet_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_wavelet_core PUBLIC numkit_value numkit_sys numkit_ops)

if(NUMKIT_WITH_SIMD)
    target_compile_definitions(numkit_wavelet_core PUBLIC NUMKIT_WITH_SIMD=1)
    target_link_libraries(numkit_wavelet_core PRIVATE hwy hwy_contrib)
endif()

if(MSVC)
    target_compile_options(numkit_wavelet_core PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_wavelet_core PRIVATE -Wall -Wextra -fPIC)
endif()

# Shared plugin module
add_library(numkit_wavelet SHARED ${WAVELET_PLUGIN_SOURCES})
add_library(numkit::wavelet ALIAS numkit_wavelet)

target_compile_definitions(numkit_wavelet PRIVATE NUMKIT_PLUGIN_BUILD=1)

target_include_directories(numkit_wavelet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_wavelet PUBLIC numkit_wavelet_core)

if(MSVC)
    target_compile_options(numkit_wavelet PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_wavelet PRIVATE -Wall -Wextra -fPIC)
endif()
