cmake_minimum_required(VERSION 3.21)

# =========================================================
# 1. 设置 Vcpkg
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project(FVM-Aerodynamic-Simulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# =========================================================
# 75=RTX 20系, 80=A100/30系前沿, 86=RTX 30系, 89=RTX 40系, 120=RTX 50系
# 最后的 "+PTX" 表示为了未来的新显卡包含虚拟架构代码(JIT编译)
set(CMAKE_CUDA_ARCHITECTURES "75;86;89;120+PTX" CACHE STRING "CUDA Archs")

# =========================================================
# 动态链接 CUDA 运行时
# 使用动态链接以匹配vcpkg库的运行时设置，避免LIBCMT冲突
# 注意：用户需要在PATH中包含CUDA DLL路径
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Find packages
find_package(OpenGL REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ImGui 设置 (保持不变)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Sources
file(GLOB PROJECT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${IMGUI_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    glad::glad
    glfw
    # 注意：CUDA::cudart_static 会自动由 CMAKE_CUDA_RUNTIME_LIBRARY 包含
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
)

# 解决LIBCMT冲突警告（如果使用静态CUDA运行时）
if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE 
        /NODEFAULTLIB:LIBCMT  # 忽略静态运行时库
    )
endif()

# =========================================================
# 构建后自动复制 assets 目录到 exe 所在目录
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
    COMMENT "正在复制 assets 目录到输出目录..."
)


# 定义一个 "安装" 目标，把 exe 和所有依赖的 DLL 放到一个文件夹里

include(GNUInstallDirs)

# 1. 安装可执行文件
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# 2. 自动抓取并安装第三方 DLL (GLFW, GLAD)
# Windows下，vcpkg 的动态库通常在 bin 目录下
include(GNUInstallDirs)

# 1. 安装 exe
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION bin
)

# 2. 【关键】自动抓取所有依赖的 DLL (GLFW, GLAD 等)
# 这里的 $<TARGET_RUNTIME_DLLS:...> 会在安装的那一刻，
# 自动去查询目标依赖了哪些 DLL，并把它们复制过来。
if(WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
        DESTINATION bin
        COMPONENT dependency
    )
endif()

# 3. 安装可能存在的资源文件 (比如 shader 或 字体)
install(DIRECTORY assets/ DESTINATION /assets)