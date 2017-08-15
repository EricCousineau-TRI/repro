
cmake_minimum_required(VERSION 2.8)
 
PROJECT(IterativeClosestPointsTransform)
 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
 
add_executable(IterativeClosestPointsTransform MACOSX_BUNDLE IterativeClosestPointsTransform.cxx )
target_link_libraries(IterativeClosestPointsTransform ${VTK_LIBRARIES})
 
add_executable(ReadWritePoly ReadWritePoly.cxx)
target_link_libraries(ReadWritePoly ${VTK_LIBRARIES})
