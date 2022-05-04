cmake_minimum_required(VERSION 3.4)
project(mf)


set(CMAKE_CXX_STANDARD 17)


find_package(CUDA)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


if(CUDA_FOUND)
	cuda_add_executable(main_gpu main.cu)
	execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "result_gpu")
endif()

add_executable(main_cpu main.cpp)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "result_cpu")
