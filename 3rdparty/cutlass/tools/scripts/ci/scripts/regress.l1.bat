:: Windows L1 (Sanity) Regression Script

setlocal
@echo on
set EXIT_CODE=0

set THIS_DIR=%~dp0

if "%ENABLE_MEMCHECK%" == "" set ENABLE_MEMCHECK=1

cmd /c %THIS_DIR%\regress.l0.bat || goto :ERROR

:EXIT

endlocal & set EXIT_CODE=%EXIT_CODE%
exit /b %EXIT_CODE%

:ERROR
if %EXIT_CODE% EQU 0 set EXIT_CODE=%ERRORLEVEL%

echo *** Error [%EXIT_CODE%]
goto :EXIT
