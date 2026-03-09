:: Windows L0 (Sanity) Regression Script

setlocal
@echo on
set EXIT_CODE=0

set THIS_DIR=%~dp0

if "%INSTALL_DIR%" == "" set INSTALL_DIR=%CD%\build\install
if NOT "%CUDA_PATH%" == "" set PATH=%CUDA_PATH%\bin;%PATH%
if "%CUTLASS_UNIT_TEST_ROOT%" == "" set "CUTLASS_UNIT_TEST_ROOT=%INSTALL_DIR%\bin"
if NOT "%CUTLASS_UNIT_TEST_FILTER%" == "" set GTEST_FILTER_ARGS="--gtest_filter=%CUTLASS_UNIT_TEST_FILTER%"
if "%ENABLE_MEMCHECK%" == "" set ENABLE_MEMCHECK=0

echo "Running cutlass_unit_test ..."

:: pushd "%CUTLASS_UNIT_TEST_ROOT%" || GOTO :ERROR

for /R "%CUTLASS_UNIT_TEST_ROOT%" %%T in (cutlass_test_unit_*.exe) do (
  echo "Run %%T ..."
  "%%T" %GTEST_FILTER_ARGS% || GOTO :ERROR
  if NOT "%ENABLE_MEMCHECK%" == "0" cuda-memcheck.exe --report-api-errors no "%%T" %GTEST_FILTER_ARGS% || GOTO :ERROR
  echo "Run %%T done."
)

:: popd || GOTO :ERROR

echo "... Done with cutlass_unit_test."

:EXIT

endlocal & set EXIT_CODE=%EXIT_CODE%
exit /b %EXIT_CODE%

:ERROR
if %EXIT_CODE% EQU 0 set EXIT_CODE=%ERRORLEVEL%

echo *** Error [%EXIT_CODE%]
goto :EXIT
