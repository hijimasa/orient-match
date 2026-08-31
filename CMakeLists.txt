cmake_minimum_required(VERSION 3.16)
project(OrientMatch
    VERSION 0.1.0
    DESCRIPTION "Dense gradient-orientation template matching for OpenCV"
    HOMEPAGE_URL "https://github.com/hijimasa/gradient-orientation-template-matching"
    LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(ORIENT_MATCH_BUILD_EXAMPLES "Build OrientMatch examples" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(OpenMP QUIET)

add_library(orient_match STATIC src/orient_match.cpp)
add_library(OrientMatch::orient_match ALIAS orient_match)
target_compile_features(orient_match PUBLIC cxx_std_17)
target_include_directories(orient_match
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(orient_match PUBLIC ${OpenCV_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(orient_match PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(orient_match PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME orient_match)

if(ORIENT_MATCH_BUILD_EXAMPLES)
    add_executable(orient_match_cli examples/orient_match_cli.cpp)
    target_link_libraries(orient_match_cli PRIVATE OrientMatch::orient_match)
endif()

if(BUILD_TESTING)
    add_executable(orient_match_tests tests/test_orient_match.cpp)
    target_link_libraries(orient_match_tests PRIVATE OrientMatch::orient_match)
    add_test(NAME orient_match_tests COMMAND orient_match_tests)
    set_tests_properties(orient_match_tests PROPERTIES TIMEOUT 120)
endif()

install(TARGETS orient_match
    EXPORT OrientMatchTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT OrientMatchTargets
    FILE OrientMatchTargets.cmake
    NAMESPACE OrientMatch::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OrientMatch)

configure_package_config_file(
    cmake/OrientMatchConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OrientMatchConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OrientMatch)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/OrientMatchConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OrientMatchConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OrientMatchConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OrientMatch)
