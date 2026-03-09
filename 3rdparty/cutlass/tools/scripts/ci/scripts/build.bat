:: Windows build
:: Prereq: Expected to be run from the root of the XMMA tree.
:: Postreq: Will create a "build" folder in which the build is executed.

setlocal
setlocal ENABLEDELAYEDEXPANSION
@echo on
set EXIT_CODE=0

if NOT DEFINED SRC_DIR set SRC_DIR=.
if NOT DEFINED BUILD_DIR set BUILD_DIR=build
if NOT DEFINED INSTALL_DIR set INSTALL_DIR=install
if NOT DEFINED CMAKE_GENERATOR set "CMAKE_GENERATOR=Visual Studio 15 2017 Win64"
if NOT DEFINED CMAKE_PLATFORM set CMAKE_PLATFORM=x64
if NOT DEFINED MSBUILD_J set MSBUILD_J=16

if NOT DEFINED SCRATCH_ROOT_DIR set SCRATCH_ROOT_DIR=c:\tmp
:: ^ For backward compatibility with older agent definitions.
if NOT DEFINED BUILD_ROOT_DIR set BUILD_ROOT_DIR=%SCRATCH_ROOT_DIR%

if NOT DEFINED CMAKE_BUILD_TYPE set CMAKE_BUILD_TYPE=Release

if DEFINED CMAKE_DIR set PATH=%CMAKE_DIR%\bin;%PATH%
if DEFINED GIT_ROOT_DIR set PATH=%GIT_ROOT_DIR%\bin;%PATH%
if DEFINED PYTHON2_DIR set PATH=%PYTHON2_DIR%\bin;%PATH%
if DEFINED PYTHON3_DIR set PATH=%PYTHON3_DIR%\bin;%PYTHON3_DIR%;%PATH%
:: ^ Sometimes there's no bin folder for Python3.

set PATH=%PATH:"=%
:: ^ Workaround for Jenkins adding quotes into the PATH variable.
:: https://issues.jenkins-ci.org/browse/JENKINS-11992?page=com.atlassian.jira.plugin.system.issuetabpanels%3Aall-tabpanel

:: Due to path length issues, we are redirecting the build into c:\Tmp. Afterward
:: we will copy the built files back into the workspace and clear the temporary directory.

set SCRIPT_DIR=%~p0
set WORK_DIR=%~f0
set WORK_DIR=%WORK_DIR:\=\\%
set WORK_DIR=%WORK_DIR:"=%

python -c "import hashlib; h = hashlib.md5(); h.update(""%WORK_DIR%"".encode(""utf-8"")); print(h.hexdigest()[0:8])" > dir.name || goto :ERROR
set /p build_suffix=<dir.name || goto :ERROR
set "SHORT_BUILD_DIR=%BUILD_ROOT_DIR%\jnk-%build_suffix%" || goto :ERROR

echo Build Tmp Directory: %SHORT_BUILD_DIR%

if EXIST "%SHORT_BUILD_DIR%\" rmdir /s /q "%SHORT_BUILD_DIR%" > NUL || call rem
mkdir "%SHORT_BUILD_DIR%" > NUL || :ERROR

if DEFINED CUDA_PATH goto :CUDA_READY
if NOT DEFINED CUDA_ARTIFACT goto :CUDA_READY
set CUDA_TOOLKITS_DIR=%BUILD_ROOT_DIR%\toolkits
mkdir "%CUDA_TOOLKITS_DIR%" || call rem

python -c "import hashlib; h = hashlib.md5(); h.update(""%CUDA_ARTIFACT%"".encode(""utf-8"")); print(h.hexdigest()[0:8])" > dir.name || goto :ERROR
set /p CUDA_ARTIFACT_HASH=<dir.name || goto :ERROR
set "CUDA_PATH=%CUDA_TOOLKITS_DIR%\%CUDA_ARTIFACT_HASH%" || goto :ERROR

if EXIST "%CUDA_PATH%\bin\nvcc.exe" goto :CUDA_READY

powershell $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri "%CUDA_ARTIFACT%" -OutFile "%CUDA_PATH%.zip" || goto :ERROR
powershell Expand-Archive "%CUDA_PATH%.zip" -DestinationPath "%CUDA_PATH%" || goto :ERROR
del "%CUDA_PATH%.zip" || call rem

:CUDA_READY

if NOT EXIST "%CUDA_PATH%\CUDAVisualStudioIntegration\extras\visual_studio_integration\" (
  echo "Visual Studio Integration not in correct place, moving ..."
  mkdir "%CUDA_PATH%\CUDAVisualStudioIntegration\extras\visual_studio_integration" || call rem
  robocopy /s /mir /mt:16 /copyall "%CUDA_PATH%\extras\visual_studio_integration" "%CUDA_PATH%\CUDAVisualStudioIntegration\extras\visual_studio_integration" || call rem
  robocopy /s /mir /mt:16 /copyall "%CUDA_PATH%\visual_studio_integration" "%CUDA_PATH%\CUDAVisualStudioIntegration\extras\visual_studio_integration" || call rem
)

dir "%CUDA_PATH%\CUDAVisualStudioIntegration\extras\visual_studio_integration\"

python %SCRIPT_DIR%\fix_cuda_toolkit.py --dir %CUDA_PATH%\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions

if DEFINED CUDA_PATH set PATH=%CUDA_PATH%\bin;%PATH%

set PATH=%PATH:"=%
:: ^ Workaround for Jenkins adding quotes into the PATH variable.
:: https://issues.jenkins-ci.org/browse/JENKINS-11992?page=com.atlassian.jira.plugin.system.issuetabpanels%3Aall-tabpanel

if DEFINED VS15_DIR call "%VS15_DIR%\Common7\Tools\VsDevCmd.bat" -host_arch=amd64 || goto :ERROR
if DEFINED VS16_DIR call "%VS16_DIR%\Common7\Tools\VsDevCmd.bat" -arch=amd64 || goto :ERROR
if DEFINED VS16_DIR set "CMAKE_GENERATOR=Visual Studio 16 2019"
if DEFINED VS17_DIR call "%VS17_DIR%\Common7\Tools\VsDevCmd.bat" -arch=amd64 || goto :ERROR
@echo on

nvcc.exe --version || goto :ERROR

set CUDA_PATH_POSIX=%CUDA_PATH:\=/%
set CUDA_PATH_POSIX=%CUDA_PATH_POSIX://=/%
set CUSTOM_CUDA_PATH_ARGS=-T cuda="%CUDA_PATH_POSIX%" -D CUDA_TOOLKIT_ROOT_DIR="%CUDA_PATH_POSIX%" -D CMAKE_CUDA_COMPILER="%CUDA_PATH_POSIX%/bin/nvcc.exe"

set CudaToolkitDir=%CUDA_PATH_POSIX%

:: Disable Git LSF fetching due to credentials issues
git lfs uninstall

:: CFK-4473 WAR on windows cmake hang
python %SCRIPT_DIR%\run_with_retries_timeout.py -r 3 -t 3500 -w %SHORT_BUILD_DIR% cmake -G "%CMAKE_GENERATOR%" -S "%SRC_DIR%" -B %SHORT_BUILD_DIR% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" %CUSTOM_CUDA_PATH_ARGS% %CMAKE_ARGS% || goto :ERROR

msbuild %SHORT_BUILD_DIR%\INSTALL.vcxproj /p:Configuration=%CMAKE_BUILD_TYPE% /m:%MSBUILD_J% /nr:false || goto :ERROR
:: Using /m to allow parallel builds, similar to make -j.
:: Using /nr:false as in GVS, MSBuild.exe
:: MSBuild.exe would keep the Visual Studio integration DLL for CUDA VS
:: integration occupied for at least 15 minuets as Nodes keep active
:: (optimization fro MS). We don't need that optimization as we have
:: requirement to build several versions with CudaToolkit on same system pool.

:FINALIZE

if DEFINED SHORT_BUILD_DIR robocopy /mir /mt:16 /np /nfl /ndl "%SHORT_BUILD_DIR%" "%BUILD_DIR%"
if ERRORLEVEL 8 if %EXIT_CODE% EQU 0 set EXIT_CODE=%ERRORLEVEL%
:: robocopy has warnings from 1-7!

if DEFINED SHORT_BUILD_DIR if exist "%SHORT_BUILD_DIR%\" (
   rmdir /s /q "%SHORT_BUILD_DIR%" > NUL || call rem
)

:EXIT

endlocal & set EXIT_CODE=%EXIT_CODE%
exit /b %EXIT_CODE%

:ERROR

if %EXIT_CODE% EQU 0 set EXIT_CODE=%ERRORLEVEL%

echo *** Error [%EXIT_CODE%]
goto :FINALIZE
