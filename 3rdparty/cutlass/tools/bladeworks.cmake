include(FetchContent)

set(BLADEWORKS_DIR "" CACHE STRING "If not empty, the location of an existing BladeWorks repository clone to use in place of an auto-fetched clone.")
set(BLADEWORKS_REVISION c1dc5298 CACHE STRING "BladeWorks Dependency Revision")

if(BLADEWORKS_DIR)
  set(FETCHCONTENT_SOURCE_DIR_BLADEWORKS ${BLADEWORKS_DIR} CACHE INTERNAL "BladeWorks local repository for override for FetchContent")
endif()

FetchContent_Declare(
  bladeworks
  GIT_REPOSITORY https://gitlab+deploy-token-232:sLWEJJBgFQCsVLypPYNu@gitlab-master.nvidia.com/dlarch-fastkernels/bladeworks.git
  GIT_TAG        ${BLADEWORKS_REVISION}
)

FetchContent_GetProperties(bladeworks)

if(NOT bladeworks_POPULATED)
  FetchContent_Populate(bladeworks)
  message(STATUS "BladeWorks: ${bladeworks_SOURCE_DIR}")
  # For now, we're just allowing clients to use the source directly.
endif()

# TODO: MOVE THE FOLLOWING INTO THE BLADEWORKS REPO

add_library(bladeworks_includes INTERFACE)

target_include_directories(
  bladeworks_includes
  INTERFACE
  $<BUILD_INTERFACE:${bladeworks_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

install(DIRECTORY ${bladeworks_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
