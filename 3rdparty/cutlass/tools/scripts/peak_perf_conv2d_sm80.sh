#
# profile cutlass operator and create performance file
#

# usage
# source ./cutlass/tools/scripts/peak_perf_conv2d.sh (runs all math instruciton by default)
# source ./cutlass/tools/scripts/peak_perf_conv2d.sh hmma_f32_f16 (takes math instruction as an optional shell argument $1) 
# Run a network on a chip for one or all of the following instructions [hmma_f32_f16, hmma_f32_tf32, imma_s32_s8, imma_s8_s8, ffma]
#

#
# Conv2d Fprop
#
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --math-instruction=${1:-all} --providers=cutlass --batch-size=34  --split-k-slices=1:6:1  --append=true --output=conv2d_fprop_resnet50_performance.csv --verification-providers=cudnn
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --math-instruction=${1:-all} --providers=cutlass --batch-size=408 --split-k-slices=1:6:1  --append=true --output=conv2d_fprop_resnet50_performance.csv
###
#######
####### profile cudnn operators and append performance file
#######
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --math-instruction=${1:-all} --kernels=cutlass_tensorop*fprop_optimized*128x128_32x* --providers=cudnn --batch-size=34  --append=true --output=conv2d_fprop_resnet50_performance.csv
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --math-instruction=${1:-all} --kernels=cutlass_tensorop*fprop_optimized*128x128_32x* --providers=cudnn --batch-size=408 --append=true --output=conv2d_fprop_resnet50_performance.csv
#
#
##
##
##
###
### Conv2d Dgrad
###
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=${1:-all} --providers=cutlass --batch-size=34  --split-k-slices=1:6:1  --append=true --output=conv2d_dgrad_resnet50_performance.csv --verification-providers=cudnn
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=${1:-all} --providers=cutlass --batch-size=408 --split-k-slices=1:6:1  --append=true --output=conv2d_dgrad_resnet50_performance.csv
###
#######
####### profile cudnn operators and append performance file (run cudnn for on single kernel instance. cudnn runs the same kernel for each cutlass instance. so don't run it again and again)
#######
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=${1:-all} --kernels=cutlass_tensorop*dgrad_analytic*128x128_32x* --providers=cudnn --batch-size=34  --append=true --output=conv2d_dgrad_resnet50_performance.csv 
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=${1:-all} --kernels=cutlass_tensorop*dgrad_analytic*128x128_32x* --providers=cudnn --batch-size=408 --append=true --output=conv2d_dgrad_resnet50_performance.csv
#
#
##
##
###
### Conv2d Wgrad
###
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --math-instruction=${1:-all} --providers=cutlass --batch-size=34  --split-k-mode=serial,parallel --split-k-slices=1:128:1 --append=true --output=conv2d_wgrad_resnet50_performance.csv --verification-providers=cudnn
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --math-instruction=${1:-all} --providers=cutlass --batch-size=408 --split-k-mode=serial,parallel --split-k-slices=1:128:1 --append=true --output=conv2d_wgrad_resnet50_performance.csv
###
#######
####### profile cudnn operators and append performance file (run cudnn for on single kernel instance. cudnn runs the same kernel for each cutlass instance. so don't run it again and again)
#######
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --math-instruction=${1:-all} --kernels=cutlass_tensorop*wgrad_optimized*128x128_32x* --providers=cudnn --batch-size=34  --append=true --output=conv2d_wgrad_resnet50_performance.csv 
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --math-instruction=${1:-all} --kernels=cutlass_tensorop*wgrad_optimized*128x128_32x* --providers=cudnn --batch-size=408 --append=true --output=conv2d_wgrad_resnet50_performance.csv
