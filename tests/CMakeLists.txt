file(GLOB TEST_SOURCES *.cpp)

add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})

target_link_libraries(${PROJECT_NAME}_tests
    PRIVATE
        ${PROJECT_NAME}_library
        gtest
        gtest_main
)

include(GoogleTest)

gtest_discover_tests(${PROJECT_NAME}_tests)