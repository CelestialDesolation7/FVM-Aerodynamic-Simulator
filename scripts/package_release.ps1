# ============================================================
# FVM-Aerodynamic-Simulator Release Packaging Script
# Usage: Run .\scripts\package_release.ps1 from project root
# ============================================================

param(
    [string]$BuildDir = "build",
    [string]$OutputDir = "dist",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  FVM-Aerodynamic-Simulator Package Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Find executable
$BuildPath = Join-Path $ProjectRoot $BuildDir
$PossibleExePaths = @(
    (Join-Path $BuildPath "$Config\FVM-Aerodynamic-Simulator.exe"),
    (Join-Path $BuildPath "FVM-Aerodynamic-Simulator.exe"),
    (Join-Path $BuildPath "bin\$Config\FVM-Aerodynamic-Simulator.exe")
)

$ExePath = $null
foreach ($path in $PossibleExePaths) {
    if (Test-Path $path) {
        $ExePath = $path
        break
    }
}

if (-not $ExePath) {
    Write-Host "[ERROR] Executable not found!" -ForegroundColor Red
    Write-Host "Please build Release version first:" -ForegroundColor Yellow
    Write-Host "  cmake -B build" -ForegroundColor Yellow
    Write-Host "  cmake --build build --config Release" -ForegroundColor Yellow
    exit 1
}

$ExeDir = Split-Path -Parent $ExePath
Write-Host "[INFO] Found executable: $ExePath" -ForegroundColor Green

# 2. Create output directory
$OutputPath = Join-Path $ProjectRoot $OutputDir
if (Test-Path $OutputPath) {
    Write-Host "[INFO] Cleaning old output directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $OutputPath
}
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
Write-Host "[INFO] Created output directory: $OutputPath" -ForegroundColor Green

# 3. Copy executable
Write-Host ""
Write-Host "[Step 1/5] Copying executable..." -ForegroundColor Cyan
Copy-Item $ExePath -Destination $OutputPath
Write-Host "  Copied: FVM-Aerodynamic-Simulator.exe" -ForegroundColor White

# 4. Copy DLLs from build directory
Write-Host ""
Write-Host "[Step 2/5] Copying DLLs from build directory..." -ForegroundColor Cyan
$DllsInBuildDir = Get-ChildItem -Path $ExeDir -Filter "*.dll" -ErrorAction SilentlyContinue
$DllCount = 0
foreach ($dll in $DllsInBuildDir) {
    Copy-Item $dll.FullName -Destination $OutputPath
    Write-Host "  Copied: $($dll.Name)" -ForegroundColor White
    $DllCount++
}
if ($DllCount -eq 0) {
    Write-Host "  [WARNING] No DLLs found in build directory" -ForegroundColor Yellow
}

# 5. Check CUDA runtime DLL
Write-Host ""
Write-Host "[Step 3/5] Checking CUDA runtime DLL..." -ForegroundColor Cyan

$CudaRuntimeDlls = Get-ChildItem -Path $OutputPath -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue
if (-not $CudaRuntimeDlls) {
    Write-Host "  [WARNING] CUDA runtime DLL not found, searching system..." -ForegroundColor Yellow
    
    $CudaPath = $env:CUDA_PATH
    if ($CudaPath -and (Test-Path $CudaPath)) {
        # CUDA 13+ stores DLLs in bin/x64, older versions in bin/
        $SearchPaths = @(
            (Join-Path $CudaPath "bin\x64"),
            (Join-Path $CudaPath "bin")
        )
        
        $CudaRuntimeSrc = $null
        foreach ($SearchPath in $SearchPaths) {
            if (Test-Path $SearchPath) {
                $CudaRuntimeSrc = Get-ChildItem -Path $SearchPath -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($CudaRuntimeSrc) {
                    Write-Host "  Found in: $SearchPath" -ForegroundColor Gray
                    break
                }
            }
        }
        
        if ($CudaRuntimeSrc) {
            Copy-Item $CudaRuntimeSrc.FullName -Destination $OutputPath
            Write-Host "  Copied from CUDA install: $($CudaRuntimeSrc.Name)" -ForegroundColor Green
        }
    }
    
    $CudaRuntimeDlls = Get-ChildItem -Path $OutputPath -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue
    if (-not $CudaRuntimeDlls) {
        Write-Host "  [ERROR] Cannot find CUDA runtime DLL!" -ForegroundColor Red
        Write-Host "  Please ensure CUDA Toolkit is installed" -ForegroundColor Red
    }
} else {
    Write-Host "  CUDA runtime DLL exists: $($CudaRuntimeDlls.Name)" -ForegroundColor Green
}

# 6. Copy assets directory
Write-Host ""
Write-Host "[Step 4/5] Copying assets..." -ForegroundColor Cyan
$AssetsSource = Join-Path $ProjectRoot "assets"
$AssetsTarget = Join-Path $OutputPath "assets"

if (Test-Path $AssetsSource) {
    Copy-Item -Path $AssetsSource -Destination $AssetsTarget -Recurse
    Write-Host "  Copied assets directory" -ForegroundColor Green
    
    $FontFile = Join-Path $AssetsTarget "fonts\msyh.ttc"
    if (Test-Path $FontFile) {
        Write-Host "  Font file included: msyh.ttc" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Font file missing" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [WARNING] assets directory not found!" -ForegroundColor Yellow
}

# 7. Create readme file
Write-Host ""
Write-Host "[Step 5/5] Creating readme file..." -ForegroundColor Cyan

$ReadmeContent = @"
# FVM Aerodynamic Simulator - Requirements

## System Requirements

### Required:
1. NVIDIA GPU (RTX 20 series or newer recommended)
   - Supported architectures: Turing, Ampere, Ada Lovelace, Blackwell
   - Driver version: 535.0 or newer recommended

2. Visual C++ Redistributable (if program fails to start):
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Recommended:
- OS: Windows 10/11 64-bit
- GPU Memory: 4GB or more
- System RAM: 8GB or more

## Troubleshooting

### Program crashes or fails to start
1. Ensure you have an NVIDIA dedicated GPU
2. Update GPU driver to latest version
3. Install Visual C++ Redistributable (link above)

### Chinese text display issues
- Ensure assets/fonts/msyh.ttc exists

### Performance issues
- Reduce grid resolution in control panel
- Disable GPU zero-copy acceleration
"@

$ReadmePath = Join-Path $OutputPath "README.txt"
$ReadmeContent | Out-File -FilePath $ReadmePath -Encoding ASCII
Write-Host "  Created: README.txt" -ForegroundColor Green

# 8. Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Package Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output directory: $OutputPath" -ForegroundColor White
Write-Host ""
Write-Host "Files included:" -ForegroundColor Yellow
Get-ChildItem -Path $OutputPath -Recurse | ForEach-Object {
    $RelPath = $_.FullName.Replace($OutputPath, "").TrimStart("\")
    if ($_.PSIsContainer) {
        Write-Host "  [DIR] $RelPath\" -ForegroundColor Cyan
    } else {
        $SizeKB = [math]::Round($_.Length / 1KB, 1)
        Write-Host "  $RelPath ($SizeKB KB)" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "[TIP] Send the entire '$OutputDir' folder to your customer" -ForegroundColor Green
Write-Host "[TIP] Customer must have NVIDIA GPU to run this program" -ForegroundColor Yellow
