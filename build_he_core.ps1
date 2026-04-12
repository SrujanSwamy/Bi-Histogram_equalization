param(
    [switch]$Clean,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Fail([string]$Message) {
    Write-Host "`nFAILED: $Message" -ForegroundColor Red
    exit 1
}

function Step([string]$Name) {
    Write-Host "`n==> $Name" -ForegroundColor Cyan
}

try {
    $projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ([string]::IsNullOrWhiteSpace($projectRoot)) {
        $projectRoot = (Get-Location).Path
    }
    Set-Location $projectRoot

    Step "Validating project layout"
    if (-not (Test-Path ".\CMakeLists.txt")) { Fail "CMakeLists.txt not found in project root." }
    if (-not (Test-Path ".\.venv\Scripts\python.exe")) { Fail ".venv Python not found. Create virtual environment first." }

    $venvPython = (Resolve-Path ".\.venv\Scripts\python.exe").Path
    $mingwGpp = "C:/msys64/ucrt64/bin/g++.exe"
    $mingwGcc = "C:/msys64/ucrt64/bin/gcc.exe"
    $mingwBin = "C:/msys64/ucrt64/bin"
    $buildDir = Join-Path $projectRoot "build"

    Step "Checking compiler/toolchain"
    if (-not (Test-Path $mingwGpp)) { Fail "MinGW g++ not found at $mingwGpp" }
    if (-not (Test-Path $mingwGcc)) { Fail "MinGW gcc not found at $mingwGcc" }

    if (-not $SkipInstall) {
        Step "Installing/ensuring Python dependencies"
        & $venvPython -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed (exit code: $LASTEXITCODE)" }
        & $venvPython -m pip install pybind11 numpy opencv-python matplotlib
        if ($LASTEXITCODE -ne 0) { Fail "pip dependency install failed (exit code: $LASTEXITCODE)" }
    } else {
        Write-Host "Skipping pip install (requested with -SkipInstall)." -ForegroundColor Yellow
    }

    Step "Resolving pybind11 CMake directory"
    $pybind11Dir = (& $venvPython -m pybind11 --cmakedir).Trim()
    $pybind11DirUnquoted = $pybind11Dir.Trim('"')
    if ([string]::IsNullOrWhiteSpace($pybind11DirUnquoted)) { Fail "Could not resolve pybind11 --cmakedir" }
    if (-not (Test-Path (Join-Path $pybind11DirUnquoted "pybind11Config.cmake"))) {
        Fail "pybind11Config.cmake not found under $pybind11DirUnquoted"
    }
    Write-Host "pybind11_DIR: $pybind11Dir"

    Step "Handling stale cache edge cases"
    if ($Clean -and (Test-Path $buildDir)) {
        Remove-Item -Recurse -Force $buildDir
        Write-Host "Removed build directory due to -Clean."
    }
    $cachePath = Join-Path $buildDir "CMakeCache.txt"
    if (Test-Path $cachePath) {
        $cacheText = Get-Content $cachePath -Raw
        $badPython = ($cacheText -match "PYTHON_EXECUTABLE:FILEPATH=C:/msys64/ucrt64/bin/python3.exe")
        $venvEscaped = [regex]::Escape($venvPython.Replace('\\', '/'))
        $wrongPython = -not ($cacheText -match "PYTHON_EXECUTABLE:FILEPATH=$venvEscaped")
        if ($badPython -or $wrongPython) {
            Remove-Item -Recurse -Force $buildDir
            Write-Host "Removed stale build cache (wrong PYTHON_EXECUTABLE)."
        }
    }

    Step "Configuring CMake"
    & cmake -S . -B build -G "MinGW Makefiles" `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_C_COMPILER="C:/msys64/ucrt64/bin/gcc.exe" `
        -DCMAKE_CXX_COMPILER="C:/msys64/ucrt64/bin/g++.exe" `
        -Dpybind11_DIR="$pybind11Dir" `
        -DPYTHON_EXECUTABLE="$venvPython"
    if ($LASTEXITCODE -ne 0) { Fail "CMake configure failed (exit code: $LASTEXITCODE)" }

    Step "Building he_core"
    & cmake --build build --config Release -j 4
    if ($LASTEXITCODE -ne 0) { Fail "CMake build failed (exit code: $LASTEXITCODE)" }

    Step "Verifying built module exists"
    $pyd = Get-ChildItem -Path $buildDir -Filter "he_core*.pyd" -ErrorAction SilentlyContinue
    if (-not $pyd) { Fail "Build completed but he_core*.pyd not found in build/." }
    $pyd | ForEach-Object { Write-Host "Built module: $($_.FullName)" }

    Step "Verifying import (without running pipeline)"
    $importCode = @"
import os
import sys
os.add_dll_directory(r""$mingwBin"")
sys.path.insert(0, os.path.abspath(""build""))
import he_core
print(""he_core import OK"")
"@
    & $venvPython -c $importCode
    if ($LASTEXITCODE -ne 0) { Fail "he_core import test failed (exit code: $LASTEXITCODE)" }

    Write-Host "`nSUCCESS: Build and import validation complete. Pipeline was not executed." -ForegroundColor Green
    Write-Host "Next (manual): python python/pipeline.py dataset/4.1.05.tiff --show-diff --diff-gain 6 --threads 4"
    exit 0
}
catch {
    Fail $_.Exception.Message
}
