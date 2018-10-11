cmake_minimum_required(VERSION 2.8)
project( fisheye )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_definitions(-std=c++11)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LIB_SRCS 变量
aux_source_directory(. DIR_LIB_SRCS)
# 生成链接库
add_library (fisheye ${DIR_LIB_SRCS} )

target_link_libraries( fisheye ${OpenCV_LIBS} )




