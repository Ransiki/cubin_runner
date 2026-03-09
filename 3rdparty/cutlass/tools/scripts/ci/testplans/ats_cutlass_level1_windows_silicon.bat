setlocal enableextensions

set CUTLASS_BUILD_DIR=%CD%
cmd /c .\tools\scripts\ci\scripts\regress.l1.bat || goto :ERROR

:EXIT
endlocal & set EXIT_CODE=%EXIT_CODE%
exit /b %EXIT_CODE%

:ERROR
if %EXIT_CODE% EQU 0 set EXIT_CODE=%ERRORLEVEL%
echo *** Error [%EXIT_CODE%]
goto :EXIT
