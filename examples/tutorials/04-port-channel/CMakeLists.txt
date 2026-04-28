add_executable(bidir_port_channel bidir_port_channel.cu)
target_link_libraries(bidir_port_channel PUBLIC mscclpp)
# target_include_directories(bidir_port_channel PUBLIC
# ${PROJECT_SOURCE_DIR}/include)
