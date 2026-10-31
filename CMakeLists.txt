cmake_minimum_required(VERSION 3.24)
project(eviacam_yunet VERSION 2.1.4 LANGUAGES C CXX RC)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT WIN32)
  message(FATAL_ERROR "This portable CMake build targets Windows. Linux users: use autotools.")
endif()
find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc imgcodecs objdetect dnn video videoio)
find_package(wxWidgets CONFIG REQUIRED)
configure_file(cmake/eviacam-config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h" COPYONLY)
add_library(face_detection src/facedetector.cpp)
target_include_directories(face_detection PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(face_detection PUBLIC ${OpenCV_LIBS})
file(GLOB APP_SOURCES CONFIGURE_DEPENDS src/*.cpp wxutil/*.cpp wxcamwindow/*.cpp)
list(REMOVE_ITEM APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/facedetector.cpp")
add_library(eviacam_resources OBJECT src/gui.rc)
target_link_libraries(eviacam_resources PRIVATE wx::core)
target_include_directories(eviacam_resources PRIVATE src)
add_executable(eviacam WIN32 ${APP_SOURCES}
  creavision/crvcamera.cpp creavision/crvcamera_cv.cpp
  creavision/crvcamera_enum.cpp creavision/crvimage.cpp creavision/crvnormroi.cpp
  simplelog/simplelog.c $<TARGET_OBJECTS:eviacam_resources>)
target_include_directories(eviacam PRIVATE ${CMAKE_CURRENT_BINARY_DIR} creavision wxutil wxcamwindow simplelog)
target_compile_definitions(eviacam PRIVATE WIN32 NOMINMAX EVIACAM_MODERN_CAPTURE UNICODE _UNICODE)
target_link_libraries(eviacam PRIVATE face_detection wx::core wx::base wx::net wx::html wx::xrc wx::adv
  comctl32 rpcrt4 winmm strmiids uuid ole32 oleaut32 shell32 ws2_32)
set_source_files_properties(src/gui.rc PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_custom_command(TARGET eviacam POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/src/resources" "$<TARGET_FILE_DIR:eviacam>/resources"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/src/data" "$<TARGET_FILE_DIR:eviacam>"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/models/face_detection_yunet_2023mar.onnx" "$<TARGET_FILE_DIR:eviacam>")
file(GLOB DOC_ASSETS CONFIGURE_DEPENDS doc/*.png doc/*.wav)
add_custom_command(TARGET eviacam POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DOC_ASSETS} "${CMAKE_CURRENT_SOURCE_DIR}/THANKS" "$<TARGET_FILE_DIR:eviacam>")
enable_testing()
add_executable(detector_tests tests/detector_tests.cpp)
target_link_libraries(detector_tests PRIVATE face_detection)
add_test(NAME detector_tests COMMAND detector_tests ".")
set_tests_properties(detector_tests PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(face_benchmark tests/face_benchmark.cpp)
target_link_libraries(face_benchmark PRIVATE face_detection)
add_executable(pipeline_tests tests/pipeline_tests.cpp src/visionpipeline.cpp src/paths.cpp
  wxutil/configbase.cpp wxutil/waittime.cpp wxutil/timeutil.cpp
  wxcamwindow/visiblenormroi.cpp creavision/crvnormroi.cpp creavision/crvimage.cpp simplelog/simplelog.c)
target_include_directories(pipeline_tests PRIVATE creavision wxutil wxcamwindow simplelog)
target_compile_definitions(pipeline_tests PRIVATE WIN32 NOMINMAX UNICODE _UNICODE)
target_link_libraries(pipeline_tests PRIVATE face_detection wx::core wx::base)
add_dependencies(pipeline_tests eviacam)
add_test(NAME pipeline_tests COMMAND pipeline_tests)
set_tests_properties(pipeline_tests PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 30)
