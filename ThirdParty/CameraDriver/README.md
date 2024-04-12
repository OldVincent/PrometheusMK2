# Camera Driver

## 适用范围

支持GalaxySDK的大恒工业相机。

## 使用方法

包含"CameraDriver.hpp"来使用该模块的函数。
可以将该驱动模块作为CMake单元加入到工程中，或者单独编译为静态库后使用头文件和静态库。

该驱动模块采取装饰器模式，CameraDevice用于且只用于控制相机设备，不具备采集功能。
采集功能由Acquisitors子命名空间下的采集器类实现。采集器类的构造函数中需要传入CameraDevice对象，从而建立采集器与相机设备的绑定关系。
一个采集器只能绑定一个CameraDevice对象，但一个CameraDevice对象却可以绑定多个采集器——虽然通常开发中不会产生这种需要。

注意，开启采集器前，要手动调用CameraDevice的打开方法。

## CUDA支持

RoboPioneers::CameraDriver::Acquisitors::GpuMatAcquisitor需要CUDA和支持CUDAd的OpenCV支持。
如果不需要使用CUDA支持，则需要在使用CameraDriver.hpp之前定义NO_CUDA宏。

## 依赖项

- GalaxySDK(C语言版)，来自[大恒图像](daheng-imaging.com)，相机驱动
- cv，OpenCV，用于承载采集的图像
- cv::cuda，即OpenCV(带有CUDA支持)，用于GpuMatAcquisitor类；定义NO_CUDA宏将关闭该类同时不再依赖该项。