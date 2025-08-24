cmake_minimum_required(VERSION 3.20)

message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")

enable_language(CUDA)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(${PROJECT_NAME}
    PRIVATE
        ano_ipo_receiver.cpp
        ano_rtp_receiver.cpp
    PUBLIC
        utils.h
        ano_ipo_receiver.h
        ano_ipo_receiver.h
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_options(${PROJECT_NAME} PUBLIC -std=c++17)
