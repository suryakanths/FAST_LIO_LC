cmake_minimum_required(VERSION 3.5)
project(aloam_velodyne)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# Enable CUDA support
option(USE_CUDA "Enable CUDA acceleration for PGO" ON)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)
        
        # Detect Jetson platform and set appropriate CUDA architectures
        if(EXISTS "/proc/device-tree/model")
            file(READ "/proc/device-tree/model" JETSON_MODEL)
            string(FIND "${JETSON_MODEL}" "NVIDIA Jetson" IS_JETSON)
            if(NOT IS_JETSON EQUAL -1)
                message(STATUS "Detected Jetson platform: ${JETSON_MODEL}")
                # Jetson CUDA architectures (covers most Jetson devices)
                # Xavier NX/AGX: 7.2, Orin: 8.7, Nano: 5.3, TX2: 6.2
                set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo --ptxas-options=-v")
                set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --default-stream per-thread")
            else()
                # Desktop GPU architectures (RTX 4060, etc.)
                set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo")
                set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
            endif()
        else()
            # Fallback for non-ARM systems (assume desktop)
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo")
            set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
        endif()
        
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA acceleration enabled for PGO")
    else()
        message(WARNING "CUDA not found, falling back to CPU-only mode")
        set(USE_CUDA OFF)
    endif()
endif()

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
  tf2_ros
  tf2_geometry_msgs
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED QUIET)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)


add_executable(ascanRegistration src/scanRegistration.cpp)
target_link_libraries(ascanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(alaserOdometry src/laserOdometry.cpp)
target_link_libraries(alaserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(alaserMapping src/laserMapping.cpp)
target_link_libraries(alaserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

# Determine PGO source files based on CUDA availability
set(PGO_SOURCES
  src/laserPosegraphOptimization.cpp
  include/scancontext/Scancontext.cpp
)

# Always include CPU fallback implementation
list(APPEND PGO_SOURCES src/pgo_cuda_utils_cpu.cc)

# Add CUDA source files if CUDA is enabled
if(USE_CUDA AND CUDA_FOUND)
    list(APPEND PGO_SOURCES 
        src/pgo_cuda_utils.cu 
        src/pgo_cuda_kernels.cu 
        src/pgo_cuda_wrapper.cc
    )
    # Enable CUDA language for this directory
    enable_language(CUDA)
endif()

add_executable(alaserPGO ${PGO_SOURCES})

# Apply OpenMP flags only to C++ files, not CUDA files
target_compile_options(alaserPGO 
  PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
)

# Base libraries
set(PGO_LIBS
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES} 
  ${CERES_LIBRARIES}
  gtsam
)

# Add OpenMP to linker flags
if(OpenMP_CXX_FOUND)
    list(APPEND PGO_LIBS ${OpenMP_CXX_LIBRARIES})
endif()

# Add CUDA libraries if enabled
if(USE_CUDA AND CUDA_FOUND)
    list(APPEND PGO_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
    )
    
    # Set CUDA properties for the executable
    set_property(TARGET alaserPGO PROPERTY CUDA_STANDARD 14)
    set_property(TARGET alaserPGO PROPERTY CUDA_STANDARD_REQUIRED ON)
    
    # Set specific compile flags for CUDA files to avoid PCL template issues
    set_source_files_properties(src/pgo_cuda_kernels.cu PROPERTIES 
        COMPILE_FLAGS "-DCUDA_SEPARABLE_COMPILATION=ON"
    )
    set_source_files_properties(src/pgo_cuda_utils.cu PROPERTIES 
        COMPILE_FLAGS "-DCUDA_SEPARABLE_COMPILATION=ON"
    )
endif()

target_link_libraries(alaserPGO ${PGO_LIBS})

add_executable(kittiHelper src/kittiHelper.cpp)
target_link_libraries(kittiHelper ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(cloud_to_2d_map src/cloud_to_2d_map.cpp)
target_link_libraries(cloud_to_2d_map ${catkin_LIBRARIES} ${PCL_LIBRARIES})




