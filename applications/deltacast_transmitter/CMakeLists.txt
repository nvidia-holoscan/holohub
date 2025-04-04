cmake_minimum_required(VERSION 3.20)
project(deltacast_transmitter_apps LANGUAGES NONE)

add_subdirectory(cpp)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()