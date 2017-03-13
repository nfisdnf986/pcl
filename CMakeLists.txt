# project
cmake_minimum_required (VERSION 2.6)
project (pcl)

#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/)

# Add to module path, so we can find our cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)

# make an uninstall target
include(${CMAKE_MODULE_PATH}/cmake_uninstall.cmake.in)
add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")


find_package(PCL REQUIRED) #  COMPONENTS common io
find_package(OpenCV REQUIRED)
find_package(Calibu REQUIRED)
find_package(HAL REQUIRED)

# directories
set (PCL_SRC_DIR src)

# include directory
#~ include_directories("${PCL_SRC_DIR}")

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-std=c++11 -msse3")

# sources
FILE(GLOB PCL_SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)
#~ set(CMAKE_BUILD_TYPE Debug)

MESSAGE( STATUS "ghost CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}" )
MESSAGE( STATUS "ghost PCL_LIBRARY_DIRS ${PCL_LIBRARY_DIRS}" )
MESSAGE( STATUS "ghost PROJECT_NAME ${PROJECT_NAME}" )

# build demo program
add_executable(${PROJECT_NAME} ${PCL_SRC_FILES})
  
include_directories("${PCL_SRC_DIR}"
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Calibu_INCLUDE_DIRS}
  ${HAL_INCLUDE_DIRS}
)

#~ link_directories(${PCL_LIBRARY_DIRS}
  #~ ${Calibu_LIBRARY}
#~ )
#~ add_definitions(${PCL_DEFINITIONS})

target_link_libraries( ${PROJECT_NAME}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${Calibu_LIBRARIES}
  ${HAL_LIBRARIES}
)
