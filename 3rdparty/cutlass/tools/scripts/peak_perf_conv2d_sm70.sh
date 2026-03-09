#
# profile cutlass operator and create performance file
#

#
# Conv2d Fprop
#
# only batch_size=1 conv2d runs are verified to save time
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=fprop --math-instruction=${1:-all} --providers=cutlass --batch-size=34 --split-k-mode=serial,parallel --split-k-slices=1:6  --append=true --output=conv2d_fprop_resnet50_performance.csv  --verification-providers=cudnn
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=fprop --math-instruction=${1:-all} --providers=cutlass --batch-size=408 --split-k-mode=serial,parallel --split-k-slices=1:6  --append=true --output=conv2d_fprop_resnet50_performance.csv
###
#######
####### profile cudnn operators and append performance file
#######
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=fprop --math-instruction=${1:-all} --kernels=cutlass_tensorop_f16_s884fprop_optimized_f16_128x128_32x2 --providers=cudnn --batch-size=34  --append=true --output=conv2d_fprop_resnet50_performance.csv 
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=fprop --math-instruction=${1:-all} --kernels=cutlass_tensorop_f16_s884fprop_optimized_f16_128x128_32x2 --providers=cudnn --batch-size=408 --append=true --output=conv2d_fprop_resnet50_performance.csv
#
#
##
##
##
###
### Conv2d Dgrad
###
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=dgrad --math-instruction=${1:-all} --providers=cutlass --batch-size=34 --split-k-mode=serial,parallel --split-k-slices=1:6  --append=true --output=conv2d_dgrad_resnet50_performance.csv --verification-providers=cudnn
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=dgrad --math-instruction=${1:-all} --providers=cutlass --batch-size=408   --split-k-mode=serial,parallel --split-k-slices=1:6  --append=true --output=conv2d_dgrad_resnet50_performance.csv
###
#######
####### profile cudnn operators and append performance file (run cudnn for on single kernel instance. cudnn runs the same kernel for each cutlass instance. so don't run it again and again)
#######
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=dgrad --math-instruction=${1:-all} --math-instruction=${1:-all} --kernels=cutlass_tensorop_f16_s884dgrad_analytic_f16_128x128_32x2 --providers=cudnn --batch-size=34  --append=true --output=conv2d_dgrad_resnet50_performance.csv 
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=dgrad --math-instruction=${1:-all} --kernels=cutlass_tensorop_f16_s884dgrad_analytic_f16_128x128_32x2 --providers=cudnn --batch-size=408 --append=true --output=conv2d_dgrad_resnet50_performance.csv
#
#
##
##
###
### Conv2d Wgrad
###
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=wgrad --math-instruction=${1:-all} --providers=cutlass --batch-size=34  --split-k-mode=serial,parallel --split-k-slices=1,127,108,86,27,54,12,10,6,13,5,3,128,20 --append=true --output=conv2d_wgrad_resnet50_performance.csv --verification-providers=cudnn
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=wgrad --math-instruction=${1:-all} --providers=cutlass --batch-size=408 --split-k-mode=serial,parallel --split-k-slices=1,127,108,86,27,54,12,10,6,13,5,3,128,20,129:156 --append=true --output=conv2d_wgrad_resnet50_performance.csv
###
#######
####### profile cudnn operators and append performance file (run cudnn for on single kernel instance. cudnn runs the same kernel for each cutlass instance. so don't run it again and again)
#######
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=wgrad --math-instruction=${1:-all} --kernels=cutlass_tensorop_f16_s884wgrad_optimized_f16_128x128_32x2 --providers=cudnn --batch-size=34  --append=true --output=conv2d_wgrad_resnet50_performance.csv 
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d  --chip="volta" --device=0 --phases=profile,process --clock=1380 --conv-kind=wgrad --math-instruction=${1:-all} --kernels=cutlass_tensorop_f16_s884wgrad_optimized_f16_128x128_32x2 --providers=cudnn --batch-size=408 --append=true --output=conv2d_wgrad_resnet50_performance.csv
