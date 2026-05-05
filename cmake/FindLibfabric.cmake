include(FindPackageHandleStandardArgs)

find_package(Libfabric CONFIG QUIET)

if(TARGET Libfabric::Libfabric)
  set(Libfabric_FOUND TRUE)
  return()
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_LIBFABRIC QUIET IMPORTED_TARGET libfabric)
  if(TARGET PkgConfig::PC_LIBFABRIC)
    add_library(Libfabric::Libfabric INTERFACE IMPORTED)
    target_link_libraries(Libfabric::Libfabric INTERFACE PkgConfig::PC_LIBFABRIC)
    set(Libfabric_FOUND TRUE)
    return()
  endif()
endif()

find_path(Libfabric_INCLUDE_DIR
  NAMES rdma/fabric.h
  HINTS ENV LIBFABRIC_ROOT
  PATH_SUFFIXES include
)

find_library(Libfabric_LIBRARY
  NAMES fabric libfabric
  HINTS ENV LIBFABRIC_ROOT
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(Libfabric
  REQUIRED_VARS Libfabric_INCLUDE_DIR Libfabric_LIBRARY
)

if(Libfabric_FOUND AND NOT TARGET Libfabric::Libfabric)
  add_library(Libfabric::Libfabric UNKNOWN IMPORTED)
  set_target_properties(Libfabric::Libfabric PROPERTIES
    IMPORTED_LOCATION "${Libfabric_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Libfabric_INCLUDE_DIR}"
  )
endif()
