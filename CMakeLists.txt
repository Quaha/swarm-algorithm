cmake_minimum_required(VERSION 3.15)

# Project name
set(PROJECT_NAME swarm-algorithm)
project(${PROJECT_NAME} LANGUAGES CXX)

# Standart version
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS include/*.h*)

add_library(${PROJECT_NAME}_library
    ${SRC_FILES} 
    ${HEADER_FILES}
)

# Header files
target_include_directories(${PROJECT_NAME}_library
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

add_executable(${PROJECT_NAME} app/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_library)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_library)

# gtest
enable_testing()
add_subdirectory(external/gtest EXCLUDE_FROM_ALL)
add_subdirectory(tests)
