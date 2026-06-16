cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOC_VERSION "${SOC_VERSION}" CACHE STRING "SOC")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "CANN")
set(RUN_MODE "npu" CACHE STRING "run mode")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
ascendc_library(kernels STATIC gelu_fp32.cpp gelu_fp16.cpp)
ascendc_include_directories(kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof)

add_library(pybind11_lib SHARED pybind11.cpp)
target_link_libraries(pybind11_lib PRIVATE kernels torch_npu m dl)

execute_process(COMMAND python3 -c
    "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_PATH)

set(ENV{ASCEND_HOME_PATH} ${ASCEND_CANN_PACKAGE_PATH})

execute_process(COMMAND python3 -c
    "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_NPU_PATH)

target_link_directories(pybind11_lib PRIVATE
    ${TORCH_PATH}/lib ${TORCH_NPU_PATH}/lib)

target_include_directories(pybind11_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_NPU_PATH}/include
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include)

set(PYBIND11_INC
    "-I/usr/local/python3.11.14/include/python3.11 -I/usr/local/python3.11.14/lib/python3.11/site-packages/torch/include")
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})
target_compile_options(pybind11_lib PRIVATE ${PYBIND11_INC} -D_GLIBCXX_USE_CXX11_ABI=1)

execute_process(COMMAND python3 -c
    "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX)

set_target_properties(pybind11_lib PROPERTIES
    OUTPUT_NAME _gelu_ext
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
