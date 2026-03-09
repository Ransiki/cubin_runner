#!/bin/tcsh
if ($#argv == 1) then
    set skip_perf_test = $argv[1]
else
    set skip_perf_test = 1
endif

echo "Test dgrad3d filter 2x2x2 stride 2x2x2 (fp16 fp32)"
echo "Test fp32"
tests.sh fp32.cu $skip_perf_test
set status = `grep -c Pass README.md`
if ( $status != 6 ) then
    echo "Error : Fp32 tests failed."
    exit -1
endif
rm -f README.md
echo "Fp32 tests pass."

echo "Test fp16"
tests.sh fp16.cu $skip_perf_test
set status = `grep -c Pass README.md`
if ( $status != 6 ) then
    echo "Error : Fp16 tests failed."
    exit -1
endif
rm -f README.md
echo "Fp16 tests pass."
echo "Test dgrad3d filter 2x2x2 stride 2x2x2 (fp16 fp32) pass."

