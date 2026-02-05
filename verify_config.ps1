# ========================================
# FVM-Aerodynamic-Simulator 配置验证脚本
# ========================================
# 用途：验证构建环境是否正确配置
# 使用：.\verify_config.ps1

Write-Host "========================================"
Write-Host "  FVM 项目环境验证"
Write-Host "========================================" -ForegroundColor Cyan

# 检查 CMake
Write-Host "`n[1/5] 检查 CMake..." -ForegroundColor Yellow
try {
    $cmakeVersion = cmake --version 2>&1 | Select-Object -First 1
    Write-Host "✓ $cmakeVersion" -ForegroundColor Green
}
catch {
    Write-Host "✗ CMake 未安装或不在 PATH 中" -ForegroundColor Red
    exit 1
}

# 检查 CUDA
Write-Host "`n[2/5] 检查 CUDA Toolkit..." -ForegroundColor Yellow
try {
    $nvccVersion = nvcc --version 2>&1 | Select-String "release"
    Write-Host "✓ CUDA: $nvccVersion" -ForegroundColor Green
}
catch {
    Write-Host "✗ CUDA Toolkit 未安装或 nvcc 不在 PATH 中" -ForegroundColor Red
    Write-Host "  请从 https://developer.nvidia.com/cuda-downloads 下载安装" -ForegroundColor Yellow
    exit 1
}

# 检查 vcpkg
Write-Host "`n[3/5] 检查 vcpkg..." -ForegroundColor Yellow
$vcpkgFound = $false
$vcpkgPaths = @(
    "$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake",
    "D:\vcpkg\scripts\buildsystems\vcpkg.cmake",
    "C:\vcpkg\scripts\buildsystems\vcpkg.cmake",
    ".\vcpkg\scripts\buildsystems\vcpkg.cmake"
)

foreach ($path in $vcpkgPaths) {
    if (Test-Path $path) {
        Write-Host "✓ 找到 vcpkg: $path" -ForegroundColor Green
        $vcpkgFound = $true
        break
    }
}

if (-not $vcpkgFound) {
    Write-Host "✗ 未找到 vcpkg" -ForegroundColor Red
    Write-Host "  请安装 vcpkg 或设置环境变量 VCPKG_ROOT" -ForegroundColor Yellow
    exit 1
}

# 检查 Visual Studio
Write-Host "`n[4/5] 检查 Visual Studio..." -ForegroundColor Yellow
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsVersion = & $vsWhere -latest -property displayName
    Write-Host "✓ $vsVersion" -ForegroundColor Green
}
else {
    Write-Host "⚠ 未找到 Visual Studio（可能使用其他编译器）" -ForegroundColor Yellow
}

# 检查项目文件
Write-Host "`n[5/5] 检查项目文件..." -ForegroundColor Yellow
$requiredFiles = @(
    "CMakeLists.txt",
    "src\main.cpp",
    "src\solver.cu",
    "external\imgui\imgui.h"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✓ $file" -ForegroundColor Green
    }
    else {
        Write-Host "✗ $file 缺失" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host "`n项目文件不完整，请检查" -ForegroundColor Red
    exit 1
}

# 总结
Write-Host "`n========================================"
Write-Host "  验证完成" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ 环境配置正确，可以开始构建" -ForegroundColor Green
Write-Host "`n下一步："
Write-Host "  1. 配置：cmake -B build -G `"Visual Studio 17 2022`" -A x64"
Write-Host "  2. 编译：cmake --build build --config Release -j"
Write-Host "  3. 运行：.\build\Release\FVM-Aerodynamic-Simulator.exe"
Write-Host "========================================"
