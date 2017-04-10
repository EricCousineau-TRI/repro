cmake_minimum_required(VERSION 3.5)
find_package(drake REQUIRED)

add_executable(main
    main.cc
    )

target_link_libraries(main
    drakeCommon)

# install(TARGETS iiwa_status_to_residual
#     RUNTIME DESTINATION bin
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib)
