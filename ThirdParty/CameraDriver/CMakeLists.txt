#==============================
# 编译要求核验
#==============================

cmake_minimum_required(VERSION 3.10)

#==============================
# 项目设定
#==============================

set(TARGET_NAME "CameraDriver")

#==============================
# 编译命令行设定
#==============================

set(CMAKE_CXX_STANDARD 17)

#==============================
# 源
#==============================

# 查找项目目录下所有源文件，记录入 TARGET_SOURCE 中
file(GLOB_RECURSE TARGET_SOURCE "*.cpp")
# 查找项目目录下所有头文件，记录入 TARGET_HEADER 中
file(GLOB_RECURSE TARGET_HEADER "*.hpp")
# 查找项目目录下所有CUDA源文件，记录入 TARGET_CUDA_SOURCE 中
file(GLOB_RECURSE TARGET_CUDA_SOURCE "*.cu")
# 查找项目目录下所有CUDA头文件，记录入 TARGET_CUDA_HEADER 中
file(GLOB_RECURSE TARGET_CUDA_HEADER "*.cuh")

#==============================
# 编译目标
#==============================

# 编译静态库
add_library(${TARGET_NAME} STATIC ${TARGET_SOURCE} ${TARGET_HEADER} ${TARGET_CUDA_SOURCE} ${TARGET_CUDA_HEADER})

#==============================
# 外部依赖
#==============================

# OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${TARGET_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PUBLIC ${OpenCV_LIBS})

# GxAPI
find_path(GX_API_INCLUDE "GxIAPI.h")
find_path(DX_IMAGE_PROC_INCLUDE "DxImageProc.h")
find_library(GX_LIB_PATH "libgxiapi.so")

target_include_directories(${TARGET_NAME} PUBLIC ${GX_API_INCLUDE} ${DX_IMAGE_PROC_INCLUDE})
target_link_libraries(${TARGET_NAME} PUBLIC ${GX_LIB_PATH})
