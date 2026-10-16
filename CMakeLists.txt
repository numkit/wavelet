cmake_minimum_required(VERSION 3.20)
project(numkit_wavelet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMKIT_WITH_SIMD "Enable Highway SIMD acceleration" ON)
option(NUMKIT_BUILD_TESTS "Build GoogleTest suite" OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_NUMKIT_TOP_BUILD_TESTS ${NUMKIT_BUILD_TESTS})
endif()

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

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NUMKIT_BUILD_TESTS ${_NUMKIT_TOP_BUILD_TESTS} CACHE BOOL "" FORCE)
endif()

if(NUMKIT_BUILD_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    if(NOT TARGET numkit)
        if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/Release/numkit.lib")
            add_library(numkit STATIC IMPORTED)
            set_target_properties(numkit PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/Release/numkit.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/include"
                INTERFACE_COMPILE_DEFINITIONS "NUMKIT_CORE_BUILD=1"
                INTERFACE_LINK_LIBRARIES "numkit_value;numkit_sys;numkit_ops"
            )
        else()
            set(NUMKIT_BUILD_CLI OFF CACHE BOOL "" FORCE)
            set(NUMKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/CMakeLists.txt")
                add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../numkit" _deps/numkit)
            else()
                FetchContent_Declare(
                    numkit_core
                    GIT_REPOSITORY https://github.com/numkit/numkit.git
                    GIT_TAG main
                )
                FetchContent_MakeAvailable(numkit_core)
            endif()
            set(NUMKIT_BUILD_TESTS ON CACHE BOOL "" FORCE)
        endif()
    endif()

    if(NOT TARGET GTest::gtest_main)
        if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest_main.lib")
            add_library(GTest::gtest STATIC IMPORTED)
            set_target_properties(GTest::gtest PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src/googletest/include"
            )
            add_library(GTest::gtest_main STATIC IMPORTED)
            set_target_properties(GTest::gtest_main PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest_main.lib"
                INTERFACE_LINK_LIBRARIES "GTest::gtest"
            )
        else()
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
            set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
            set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src/CMakeLists.txt")
                set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src" CACHE PATH "" FORCE)
            endif()
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.14.0
            )
            FetchContent_MakeAvailable(googletest)
        endif()
    endif()

    file(GLOB WAVELET_TEST_SOURCES "tests/*_test.cpp")
    add_executable(numkit_wavelet_test ${WAVELET_TEST_SOURCES})
    target_include_directories(numkit_wavelet_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(numkit_wavelet_test PRIVATE
        numkit_wavelet_core
        numkit
        GTest::gtest_main
    )
    if(NUMKIT_WITH_SIMD)
        target_link_libraries(numkit_wavelet_test PRIVATE hwy hwy_contrib)
    endif()
    add_dependencies(numkit_wavelet_test numkit_wavelet)
    if(MSVC)
        target_compile_options(numkit_wavelet_test PRIVATE /W3 /MP /bigobj)
    endif()
    add_test(NAME numkit_wavelet_test COMMAND numkit_wavelet_test)
endif()
