#==============================
# 编译要求核验
#==============================

cmake_minimum_required(VERSION 3.10)

#==============================
# 项目设定
#==============================

set(TARGET_NAME "SparrowEngine")

#==============================
# 编译命令行设定
#==============================

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(
            -DDEBUG
    )
endif()

#==============================
# 源
#==============================

# 查找项目目录下所有源文件，记录入 TARGET_SOURCE 中
file(GLOB_RECURSE TARGET_SOURCE "*.cpp")
# 查找项目目录下所有头文件，记录入 TARGET_HEADER 中
file(GLOB_RECURSE TARGET_HEADER "*.hpp")
# 查找项目目录下所有CUDA源文件，记录入 TARGET_CUDA_SOURCE 中
file(GLOB_RECURSE TARGET_CUDA_SOURCE "*.cu")
# 查找项目目录下所有CUDA源文件，记录入 TARGET_CUDA_HEADER 中
file(GLOB_RECURSE TARGET_CUDA_HEADER "*.cuh")

#==============================
# 编译目标
#==============================

# 编译静态库
add_library(${TARGET_NAME} STATIC ${TARGET_SOURCE} ${TARGET_HEADER} ${TARGET_CUDA_SOURCE} ${TARGET_CUDA_HEADER})

#==============================
# 外部依赖
#==============================

# 外部模块目录
target_include_directories(${TARGET_NAME} PUBLIC "../ThirdParty/")

# 相机驱动
target_link_libraries(${TARGET_NAME} PUBLIC "CameraDriver")

# 串口驱动
target_link_libraries(${TARGET_NAME} PUBLIC "SerialPortDriver")

# OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${TARGET_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PUBLIC ${OpenCV_LIBS})

# Boost
find_package(Boost 1.71 REQUIRED COMPONENTS system thread filesystem program_options)
target_include_directories(${TARGET_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PUBLIC ${Boost_LIBRARIES})