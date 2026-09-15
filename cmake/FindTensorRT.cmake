set(_TensorRT_HINTS)
if(TensorRT_ROOT)
    list(APPEND _TensorRT_HINTS "${TensorRT_ROOT}")
endif()
if(DEFINED ENV{TensorRT_ROOT})
    list(APPEND _TensorRT_HINTS "$ENV{TensorRT_ROOT}")
endif()

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES include include/x86_64-linux-gnu
)

find_library(TensorRT_NVINFER_LIBRARY
    NAMES nvinfer
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu
)

find_library(TensorRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY TensorRT_PLUGIN_LIBRARY
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
    set(TensorRT_LIBRARIES "${TensorRT_NVINFER_LIBRARY}" "${TensorRT_PLUGIN_LIBRARY}")
    get_filename_component(_TensorRT_LIB_DIR "${TensorRT_NVINFER_LIBRARY}" DIRECTORY)
    set(TensorRT_LIB_DIRS "${_TensorRT_LIB_DIR}")
endif()

mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIBRARY
    TensorRT_PLUGIN_LIBRARY
)
