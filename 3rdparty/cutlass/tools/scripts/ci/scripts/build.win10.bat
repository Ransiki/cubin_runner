:: Windows build
:: Prereq: Expected to be run from the root of the CUTLASS tree.
:: Postreq: Will create a "build" folder in which the build is executed.

setlocal
setlocal ENABLEDELAYEDEXPANSION
@echo on
set EXIT_CODE=0
set THIS_DIR=%~dp0

call %THIS_DIR%\build.bat || GOTO :ERROR

:EXIT

endlocal & set EXIT_CODE=%EXIT_CODE%
exit /b %EXIT_CODE%

:ERROR

if %EXIT_CODE% EQU 0 set EXIT_CODE=%ERRORLEVEL%

echo *** Error [%EXIT_CODE%]
goto :EXIT
