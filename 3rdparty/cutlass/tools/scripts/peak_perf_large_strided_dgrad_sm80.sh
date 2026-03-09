# CUTLASS large strided dgrad 
#python3  ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d --tags=${1:-branch:dev} --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=hmma_f32_f16 --network-name=LargeStridedLayers --kernels=dgrad_optimized --providers=cutlass  --append=true --output=conv2d_dgrad_resnet50_performance.csv --verification-providers=cudnn
python3  ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d --tags=${1:-branch:dev} --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=hmma_f32_f16 --network-name=LargeStridedLayers --kernels=dgrad_analytic --providers=cutlass  --append=true --output=conv2d_dgrad_resnet50_performance.csv --verification-providers=cudnn


# cuDNN large strided dgrad 
python3  ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d --tags=${1:-branch:dev} --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=hmma_f32_f16 --network-name=LargeStridedLayers --kernels=cutlass_tensorop*dgrad_analytic*128x128_32x5 --providers=cudnn --append=true --output=conv2d_dgrad_resnet50_performance.csv 