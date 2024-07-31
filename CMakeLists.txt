# Do not change these options in this file. Use cmake.config, cmake -DOPTION=VALUE, or ccmake to specify them.
option(BUILD_TEST "Build tests" OFF)

cmake_minimum_required(VERSION 3.17)

include(CMakeDependentOption)
project(InfiniTensor C CXX)

cmake_dependent_option(BUILD_TEST_CORE "Build tests for core components" ON BUILD_TEST OFF)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
# Build Type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Configuring for Debug build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    add_compile_definitions(DEBUG_MODE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Configuring for Release build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    add_compile_definitions(NDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message("Configuring for RelWithDebInfo build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")
else()
    message("Build type not specified. Configuring for RelWithDebInfo build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=gnu++11 when on, -std=c++11 when off
add_compile_options(-Wno-error=unused-variable)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED)

# OpenMP
find_package(OpenMP)
if(OpenMP_C_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(include)

if(BUILD_TEST)
  set(BUILD_GMOCK
      OFF
      CACHE BOOL "Do not build gmock" FORCE)
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "Do not install gtest" FORCE)
  add_subdirectory(3rd-party/googletest)
  include_directories(3rd-party/googletest/googletest/include)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Werror -Wno-error=deprecated-declarations -Wno-error=pointer-arith")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG") # Enable assertion
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -UNDEBUG") # Enable assertion


# Source files
file(GLOB_RECURSE SRC src/core/*.cc src/kernels/cpu/*.cc src/operators/*.cc src/utils/*.cc)

if(USE_INTELCPU)
  file(GLOB_RECURSE SRC_INTELCPU src/intelcpu/*.cc src/kernels/intelcpu/*.cc )
  list (APPEND SRC ${SRC_INTELCPU})
endif()

# Libraries
add_library(InfiniTensor SHARED ${SRC})

function(build_test files)
  # Non-recursive glob for skip failed tests
  file(GLOB TEST_SOURCES ${files})
  foreach(testsourcefile ${TEST_SOURCES})
    get_filename_component(testname ${testsourcefile} NAME_WE)
    add_executable(${testname} ${testsourcefile})
    target_link_libraries(${testname} InfiniTensor GTest::gtest_main)
    add_test(NAME ${testname} COMMAND ${testname})
  endforeach(testsourcefile ${TEST_SOURCES})
endfunction()

if(BUILD_TEST)
  add_compile_definitions(BUILD_TEST=1)
  enable_testing()
  if(BUILD_TEST_CORE)
    build_test(test/core/*.cc)
    build_test(test/operators/*.cc)
    build_test(test/kernels/nativecpu/*.cc)
  endif()
endif()
