#==============================
# 编译要求核验
#==============================

cmake_minimum_required(VERSION 3.10)

#==============================
# 项目设定
#==============================

project("Project Prometheus Mk2" LANGUAGES CXX CUDA)


#==============================
# 内部编译单元
#==============================

add_subdirectory("SparrowEngine")
add_subdirectory("Prometheus")

#==============================
# 外部编译单元
#==============================

add_subdirectory("ThirdParty/CameraDriver")
add_subdirectory("ThirdParty/SerialPortDriver")