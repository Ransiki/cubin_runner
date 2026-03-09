:: Windows build
:: Prereq: Expected to be run from the root of the CUTLASS tree.
:: Postreq: Will create a "build" folder in which the build is executed.

setlocal
@echo on
set EXIT_CODE=0

set THIS_DIR=%~dp0

set CMAKE_ARGS=-D CUTLASS_NVCC_ARCHS="70;75" -D CUTLASS_ENABLE_EXTENDED_PTX=ON -D CUTLASS_ENABLE_INTERNAL_NVVM=ON -DCUTLASS_ENABLE_CUDNN=ON %CMAKE_ARGS%

call %THIS_DIR%\build.win10.bat || goto :ERROR

:EXIT

endlocal & set EXIT_CODE=%EXIT_CODE%
exit /b %EXIT_CODE%

:ERROR

if %EXIT_CODE% EQU 0 set EXIT_CODE=%ERRORLEVEL%

echo *** Error [%EXIT_CODE%]
goto :EXIT
