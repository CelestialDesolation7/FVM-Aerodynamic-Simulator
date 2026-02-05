cmake_minimum_required(VERSION 3.21)

# =========================================================
# 1. 设置 Vcpkg
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

# =========================================================
# 2. 指定 CUDA 版本（用于兼容性）
# 如果需要兼容 CUDA 12.x 驱动，请安装 CUDA 12 Toolkit 并设置此路径
# 可以通过命令行覆盖：cmake -B build -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
if(NOT DEFINED CUDAToolkit_ROOT)
    # 优先使用 CUDA 12.x（更好的兼容性）
    set(CUDA12_PATHS
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.7"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
    )
    foreach(CUDA_PATH ${CUDA12_PATHS})
        if(EXISTS "${CUDA_PATH}")
            set(CUDAToolkit_ROOT "${CUDA_PATH}" CACHE PATH "CUDA Toolkit Root")
            set(CMAKE_CUDA_COMPILER "${CUDA_PATH}/bin/nvcc.exe" CACHE FILEPATH "CUDA Compiler")
            message(STATUS "Using CUDA 12.x for better compatibility: ${CUDA_PATH}")
            break()
        endif()
    endforeach()
endif()

project(FVM-Aerodynamic-Simulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# =========================================================
# GPU 架构设置
# 75=RTX 20系, 86=RTX 30系, 89=RTX 40系
# 注意：120(RTX 50系) 需要 CUDA 13+，为了兼容 CUDA 12 这里不包含
# "+PTX" 表示包含虚拟架构代码，支持未来显卡的JIT编译
set(CMAKE_CUDA_ARCHITECTURES "75;86;89+PTX" CACHE STRING "CUDA Archs")

# =========================================================
# 静态链接 CUDA 运行时（推荐用于分发）
# 这样客户机不需要安装 CUDA，也不需要分发 cudart DLL
# 只要客户的显卡驱动支持所需的 CUDA 版本即可
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

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

# 解决运行时库冲突警告
if(MSVC)
    # 静态 CUDA 运行时会引入 LIBCMT，需要忽略以避免与 vcpkg 的动态运行时冲突
    target_link_options(${PROJECT_NAME} PRIVATE 
        /NODEFAULTLIB:LIBCMT
        /NODEFAULTLIB:LIBCMTD
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

# =========================================================
# 【关键修复】构建后自动复制所有依赖 DLL 到 exe 目录
# 这确保直接运行 exe 时不会因缺少 DLL 而闪退
if(WIN32)
    # 1. 复制 CMake 追踪的运行时 DLL (vcpkg 的 GLFW, GLAD 等)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND_EXPAND_LISTS
        COMMENT "正在复制 vcpkg 依赖 DLL..."
    )
    
    # 2. 复制 CUDA 运行时 DLL (cudart64_*.dll)
    # CMAKE_CUDA_RUNTIME_LIBRARY=Shared 模式下必须分发这些 DLL
    # 注意：CUDA 13+ 的 DLL 位于 bin/x64 子目录中
    if(CUDAToolkit_FOUND)
        get_filename_component(CUDA_BIN_DIR "${CUDAToolkit_BIN_DIR}" ABSOLUTE)
        
        # 搜索多个可能的路径 (CUDA 12 及之前在 bin/, CUDA 13+ 在 bin/x64/)
        file(GLOB CUDA_RUNTIME_DLLS 
            "${CUDA_BIN_DIR}/cudart64_*.dll"
            "${CUDA_BIN_DIR}/x64/cudart64_*.dll"
        )
        
        if(CUDA_RUNTIME_DLLS)
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CUDA_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
                COMMENT "正在复制 CUDA 运行时 DLL..."
            )
        else()
            message(WARNING "未找到 CUDA 运行时 DLL，请手动复制 cudart64_*.dll 到输出目录")
        endif()
    endif()
endif()

# =========================================================
# 定义一个 "安装" 目标，用于创建完整的可分发包

include(GNUInstallDirs)

# 设置安装前缀（默认为 dist 目录）
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/dist" CACHE PATH "安装路径" FORCE)
endif()

# 1. 安装可执行文件
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION .
)

# 2. 自动抓取并安装第三方 DLL (GLFW, GLAD 等)
if(WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
        DESTINATION .
        COMPONENT dependency
    )
    
    # 3. 安装 CUDA 运行时 DLL (支持 CUDA 12 和 13+)
    if(CUDAToolkit_FOUND)
        get_filename_component(CUDA_BIN_DIR "${CUDAToolkit_BIN_DIR}" ABSOLUTE)
        file(GLOB CUDA_RUNTIME_DLLS_INSTALL 
            "${CUDA_BIN_DIR}/cudart64_*.dll"
            "${CUDA_BIN_DIR}/x64/cudart64_*.dll"
        )
        if(CUDA_RUNTIME_DLLS_INSTALL)
            install(FILES ${CUDA_RUNTIME_DLLS_INSTALL}
                DESTINATION .
                COMPONENT cuda_runtime
            )
        endif()
    endif()
endif()

# 4. 安装资源文件 (字体等)
install(DIRECTORY assets/ DESTINATION assets)