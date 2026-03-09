#
# profile cutlass operator and create performance file
#

# usage
# source ./cutlass/tools/scripts/peak_perf_conv2d.sh (runs all math instruciton by default)
# source ./cutlass/tools/scripts/peak_perf_conv2d.sh hmma_f16_f16 (takes math instruction as an optional shell argument $1) 
# Run a network on a chip for one or all of the following instructions [hmma_f16_f16, imma_s8_s8_nhwc, imma_s8_s8_ncxhwx, imma_s4_s4_nhwc, imma_s4_s4_ncxhwx]
#

#
# Conv2d Fprop (SM80 device=0)
#
# provider = CUTLASS
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --device=0 --phases=profile,process --clock=1005 --conv-kind=fprop --math-instruction=${1:-all} --providers=cutlass --batch-size=34  --split-k-slices=1:6:1  --append=true --output=conv2d_fprop_resnet50_performance.csv 

# provider = cuDNN (for cudnn inference only run (F16 <= F16*F16+F16). IMMAs are not cleanly supported by cuDNN)
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --device=0 --phases=profile,process --clock=1005 --conv-kind=fprop --math-instruction=hmma_f16_f16 --kernels=cutlass_tensorop*fprop_optimized*128x128_32x* --providers=cudnn --batch-size=34  --append=true --output=conv2d_fprop_resnet50_performance.csv


#
# Conv2d Fprop (SM75 device=1)
#
# provider = CUTLASS
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="turing" --device=1 --phases=profile,process --clock=1320 --conv-kind=fprop --math-instruction=${1:-all} --providers=cutlass --batch-size=34  --split-k-slices=1:6:1  --append=true --output=conv2d_fprop_resnet50_performance.csv 

# provider = (for cudnn inference only run (F16 <= F16*F16+F16). IMMAs are not cleanly supported by cuDNN)
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="turing" --device=1 --phases=profile,process --clock=1320 --conv-kind=fprop --math-instruction=hmma_f16_f16 --kernels=cutlass_tensorop*fprop_optimized*128x128_32x* --providers=cudnn --batch-size=34  --append=true --output=conv2d_fprop_resnet50_performance.csv 
