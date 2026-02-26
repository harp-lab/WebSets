cmake_minimum_required(VERSION 3.0...3.25)
project(SetOperationsByKeyGPU LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Avoid MAX_PATH issues on Windows
set(FETCHCONTENT_BASE_DIR "D:/fc" CACHE PATH "FetchContent base directory")

include(FetchContent)
FetchContent_Declare(
    webgpu
    GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
    GIT_TAG        wgpu-v24.0.0.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(webgpu)

add_executable(App main.cpp)
target_link_libraries(App PRIVATE webgpu)

# Copy wgpu DLL next to executable
target_copy_webgpu_binaries(App)

# Copy shader files to build output directory
set(SHADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/balanced_path_biased.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/set_availability_decoupled_lookback_by_key.wgsl
)

foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    configure_file(${SHADER} ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME} COPYONLY)
endforeach()

# Also copy shaders next to the executable for multi-config generators (MSVC)
add_custom_command(TARGET App POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:App>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SHADER_FILES}
        "$<TARGET_FILE_DIR:App>/shaders/"
    COMMENT "Copying shader files to output directory"
)
