# CUTLASS strided dgrad (batch-size 34)

# optimized
python3  ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d --tags=${1:-branch:dev} --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=hmma_f32_f16 --network-name=StridedLayers --kernels=dgrad_optimized --providers=cutlass --batch-size=34  --append=true --output=conv2d_dgrad_resnet50_performance.csv --verification-providers=cudnn

# analytic
#python3  ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d --tags=${1:-branch:dev} --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=hmma_f32_f16 --network-name=StridedLayers --kernels=dgrad_analytic --providers=cutlass --batch-size=34  --append=true --output=conv2d_dgrad_resnet50_performance.csv --verification-providers=cudnn


# cuDNN strided dgrad (batch-size 34)
python3  ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv2d --tags=${1:-branch:dev} --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --math-instruction=hmma_f32_f16 --network-name=StridedLayers --kernels=cutlass_tensorop*dgrad_optimized*128x128_32x5 --providers=cudnn --batch-size=34  --append=true --output=conv2d_dgrad_resnet50_performance.csv 
