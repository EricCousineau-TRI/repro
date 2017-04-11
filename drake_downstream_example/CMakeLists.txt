cmake_minimum_required(VERSION 3.5)
find_package(drake REQUIRED)
add_executable(main
    main.cc
    )
target_link_libraries(main
    drake::drake)
