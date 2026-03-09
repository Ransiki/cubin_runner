# 
# Script to produce cutlass gemm performance results against cuda toolkit version ($1). 
# The script expects cuda toolkit version to be provided as its only argument.
#

#
# Usage
#

# generate gemm kernels : `cmake .. -DCUTLASS_NVCC_ARCHS='80' -DCUTLASS_LIBRARY_KERNELS="1688tf32gemm*align4,i16832gemm*align16,s16816gemm_f16*align8,d884gemm*align1" -DJETFIRE_ENABLED=1`

# compile               : `make cutlass_profiler -j12`

# lock clock            : `sudo nvidia-smi -pm 1 -i 0`
#                         `sudo nvidia-smi -i 0 -lgc 1005,1005`  [GA100]
#                         `sudo nvidia-smi -i 0 -lgc 1335,1335`  [TU102]
#                         `sudo nvidia-smi -i 0 -lgc 1380,1380`  [GV100]
#                         

# lock power            : `sudo nvidia-smi -pl 400 -i 0` [GA100]

# run                   : `source ../cutlass/tools/scripts/run_gemm_regression.sh [compiler_version]`
# example run           : `source ../cutlass/tools/scripts/run_gemm_regression.sh 11_0_194`  

#HMMA.1688.F32.TF32
./tools/profiler/cutlass_profiler --m=8192 --n=6912 --k=12288 --clock=1005 --providers=cutlass --profiling-iterations=10  --kernels=1688tf32gemm --output=$1.csv --tags=cuda_toolkit:$1,math_instruction:HMMA.1688.F32.TF32

#IMMA.16832.S32.S8
./tools/profiler/cutlass_profiler --m=8192 --n=6912 --k=49152 --clock=1005 --providers=cutlass --profiling-iterations=10 --kernels=i16832gemm --output=$1.csv --append=true --tags=cuda_toolkit:$1,math_instruction:IMMA.16832.S32.S8

#HMMA.16816.*.F16
./tools/profiler/cutlass_profiler --m=8192 --n=6912 --k=24576 --clock=1005 --providers=cutlass --profiling-iterations=10 --kernels=s16816gemm --output=$1.csv --append=true --tags=cuda_toolkit:$1,math_instruction:HMMA.16816.*.F16

#DMMA.884.F64.F64
./tools/profiler/cutlass_profiler --m=8192 --n=6912 --k=16384 --clock=1005 --providers=cutlass --profiling-iterations=4 --kernels=d884gemm --output=$1.csv --append=true --tags=cuda_toolkit:$1,math_instruction:DMMA.884.F64.F64