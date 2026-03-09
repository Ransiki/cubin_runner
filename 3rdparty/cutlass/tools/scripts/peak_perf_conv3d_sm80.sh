#
# profile cutlass operator and create performance file
#

# usage
# source ./cutlass/tools/scripts/peak_perf_conv3d.sh (runs VNet layers by default)
# source ./cutlass/tools/scripts/peak_perf_conv3d.sh Resnet50 (takes network as an optional shell argument $1)

#
# conv3d Fprop
#
# only batch_size=1 conv3d runs are verified to save time
#python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --providers=cutlass --batch-size=1   --split-k-slices=1:6:1  --append=true --output=conv3d_fprop_${1:-VNet}_performance.csv --verification-providers=cudnn
#python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --providers=cutlass --batch-size=34  --split-k-slices=1:6:1  --append=true --output=conv3d_fprop_${1:-VNet}_performance.csv
#python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --providers=cutlass --batch-size=408 --split-k-slices=1:6:1  --append=true --output=conv3d_fprop_${1:-VNet}_performance.csv
###
#######
####### profile cudnn operators and append performance file
#######
##python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --providers=cudnn --batch-size=1   --append=true --output=conv3d_fprop_${1:-VNet}_performance.csv
#python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --kernels=cutlass_tensorop_*fprop_optimized*128x128_32x* --providers=cudnn --batch-size=34  --append=true --output=conv3d_fprop_${1:-VNet}_performance.csv 
#python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=fprop --kernels=cutlass_tensorop_*fprop_optimized*128x128_32x* --providers=cudnn --batch-size=408 --append=true --output=conv3d_fprop_${1:-VNet}_performance.csv
#
#
##
##
##
###
### conv3d Dgrad
###
### only batch_size=1 conv3d runs are verified to save time
###python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --providers=cutlass --batch-size=1   --split-k-slices=1:6:1  --append=true --output=conv3d_dgrad_${1:-VNet}_performance.csv --verification-providers=cudnn
#python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --providers=cutlass --batch-size=34  --split-k-slices=1:6:1  --append=true --output=conv3d_dgrad_${1:-VNet}_performance.csv
#python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --providers=cutlass --batch-size=408 --split-k-slices=1:6:1  --append=true --output=conv3d_dgrad_${1:-VNet}_performance.csv
###
#######
####### profile cudnn operators and append performance file (run cudnn for on single kernel instance. cudnn runs the same kernel for each cutlass instance. so don't run it again and again)
#######
###python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --providers=cudnn --batch-size=1   --append=true --output=conv3d_dgrad_${1:-VNet}_performance.csv
#python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --kernels=cutlass_tensorop_*dgrad_optimized*128x128_32x* --providers=cudnn --batch-size=34  --append=true --output=conv3d_dgrad_${1:-VNet}_performance.csv 
#python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=dgrad --kernels=cutlass_tensorop_*dgrad_optimized*128x128_32x* --providers=cudnn --batch-size=408 --append=true --output=conv3d_dgrad_${1:-VNet}_performance.csv
#
#
##
##
###
### conv3d Wgrad
###
### only batch_size=1 conv3d runs are verified to save time
###python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --providers=cutlass --batch-size=1  --split-k-slices=1:6:1 --append=true --output=conv3d_wgrad_${1:-VNet}_performance.csv --verification-providers=cudnn
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --providers=cutlass --batch-size=2  --split-k-mode=serial,parallel --split-k-slices=1:128 --append=true --output=conv3d_wgrad_${1:-VNet}_performance.csv --verification-providers=cudnn
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --providers=cutlass --batch-size=256  --split-k-mode=serial,parallel --split-k-slices=1:128 --append=true --output=conv3d_wgrad_${1:-VNet}_performance.csv  --verification-providers=cudnn
###
#######
####### profile cudnn operators and append performance file (run cudnn for on single kernel instance. cudnn runs the same kernel for each cutlass instance. so don't run it again and again)
#######
###python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --kernels=cutlass_tensorop_*wgrad3d_optimized*128x128_32x* --providers=cudnn --batch-size=1   --append=true --output=conv3d_wgrad_${1:-VNet}_performance.csv
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --kernels=cutlass_tensorop_*wgrad3d_optimized*128x128_32x* --providers=cudnn --batch-size=2  --append=true --output=conv3d_wgrad_${1:-VNet}_performance.csv 
python3 ../cutlass/tools/scripts/peak_perf_conv.py --operation=conv3d --network-name=${1:-VNet}  --chip="ampere" --phases=profile,process --clock=1005 --conv-kind=wgrad --kernels=cutlass_tensorop_*wgrad3d_optimized*128x128_32x* --providers=cudnn --batch-size=256  --append=true --output=conv3d_wgrad_${1:-VNet}_performance.csv 
#