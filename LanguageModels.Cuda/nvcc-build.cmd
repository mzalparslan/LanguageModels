@echo off
rem Compiles one .cu file to an object file with nvcc, for the LanguageModels.Cuda project.
rem
rem Why a script: the CUDA Toolkit's Visual Studio integration is not installed for
rem every Visual Studio, and nvcc needs the MSVC build environment (cl.exe and its
rem include/lib paths) that plain MSBuild does not set up for a custom build step.
rem This script builds that environment for the right MSVC toolset, then runs nvcc.
rem
rem Usage:
rem   nvcc-build.cmd <source.cu> <output.obj> <Debug|Release> <MSVC toolset version>
rem                  <VC install dir> <solution dir> <compute capability, e.g. 89>
rem
rem The toolset version must be one the installed CUDA Toolkit accepts as its host
rem compiler (CUDA 13.0 rejects MSVC 19.50 and newer, so the project uses v143).
setlocal

set "SOURCE=%~1"
set "OUTPUT=%~2"
set "CONFIG=%~3"
set "TOOLSET=%~4"
set "VCDIR=%~5"
set "SOLUTION=%~6"
set "SM=%~7"

if "%SM%"=="" (
    echo nvcc-build: missing arguments.
    exit /b 1
)
if not defined CUDA_PATH (
    echo nvcc-build: CUDA_PATH is not set. Install the CUDA Toolkit.
    exit /b 1
)

call "%VCDIR%Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%TOOLSET% >nul
if errorlevel 1 (
    echo nvcc-build: could not set up the MSVC %TOOLSET% environment.
    exit /b 1
)

rem Same C runtime as the projects that link the result: /MDd in Debug, /MD in Release.
if /i "%CONFIG%"=="Debug" (
    set "CONFIGFLAGS=-G -Xcompiler /MDd,/Od"
) else (
    set "CONFIGFLAGS=-O2 -lineinfo -Xcompiler /MD,/O2"
)

"%CUDA_PATH%\bin\nvcc.exe" -c "%SOURCE%" -o "%OUTPUT%" ^
    -std=c++20 ^
    -gencode arch=compute_%SM%,code=sm_%SM% ^
    -gencode arch=compute_%SM%,code=compute_%SM% ^
    %CONFIGFLAGS% ^
    -Xcompiler /W3 ^
    -I "%SOLUTION%LanguageModels\include\cuda" ^
    -I "%SOLUTION%LanguageModels\include\common"
exit /b %errorlevel%
