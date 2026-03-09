import json

###########################################################
##
##  Variables/Fields that can be configured by the user
##
###########################################################

##### Fields most likely to be modified
# output path
run_dir = "/home/scratch.prramani_gpu_4/cutlass_perfsim_runs/"

# Additional branch to pull from before building cutlass
cutlass_pull_branch = "dev"

# Update this for a custom driver version
cudacxx = "/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/bin/nvcc"
ld_library_path = "/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/lib64" 

# To run your custom binary edit this portion - Refer gemm_binary and gemm_testlist variables below
# your_binary = "path_to_binary_relative_to_cutlass_repo_base"
# your_testlist = ["gtest_filter_arg1", "gtest_filter_arg2", ...]
# binary_list = [your_binary]
# testlist = [your_testlist]

# This variable must be a list (edit this to run your custom binary)
binary_list = [None]
test_list = [None]

###### Fields less likely to be modified
project = "GA100_CUTLASS_PERFSIM"
cutlass_dev = "ssh://git@gitlab-master.nvidia.com:12051/dlarch-fastkernels/cutlass.git"
arch = 82
flow_config_file = "/home/scratch.prramani_gpu_4/cutlass_perfsim_runs/perfsim-runner/config.perfsim.yml"
flow_perfsim_path = "/home/scratch.svc_compute_arch/release/flow.perfsim/latest/flow_perfsim/flow.perfsim"

# Update this to run a different set of gemm tests
gemm_binary = "build/test/unit/gemm/device/perfsim/cutlass_gemm_device_perfsim"
gemm_testlist = [
"SM80_Device_Gemm.FP16in_FP32acc_NT_256x128x64_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_256x128x64_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_256x128x32_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_256x128x32_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_256x64x64_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_256x64x64_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_256x64x32_64x64x16_4stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_256x64x32_64x64x16_4stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_128x256x64_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_128x256x64_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_128x256x32_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_128x256x32_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_128x128x64_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_128x128x64_64x64x16_3stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_128x128x32_64x64x16_4stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_128x128x32_64x64x16_4stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_128x64x64_64x32x16_4stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_128x64x64_64x32x16_4stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_128x64x32_64x32x16_6stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_128x64x32_64x32x16_6stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_64x128x32_32x64x16_6stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_64x128x32_32x64x16_6stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_64x128x64_32x64x16_4stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_64x128x64_32x64x16_4stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_64x64x64_32x32x16_6stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_64x64x64_32x32x16_6stage",
"SM80_Device_Gemm.FP16in_FP32acc_NT_64x64x32_32x32x16_10stage",
"SM80_Device_Gemm.FP16in_FP32acc_TN_64x64x32_32x32x16_10stage",
"SM80_Device_Gemm.FP64in_FP64acc_TN_128x128x16_32x64x4",
"SM80_Device_Gemm.FP64in_FP64acc_NT_128x128x16_32x64x4",
"SM80_Device_Gemm_s8n_s8t_s8n_tensor_op_s32_perfsim.128x128x64_64x64x32",
"SM80_Device_Gemm_s8n_s8t_s8n_tensor_op_s32_perfsim.256x128x64_64x64x32",
"SM80_Device_Gemm_s8n_s8t_s8n_tensor_op_s32_perfsim.128x256x64_64x64x32",
"SM80_Device_Gemm_s8t_s8n_s32n_tensor_op_s32_perfsim.128x256x128_64x64x32",
"SM80_Device_Gemm_s8t_s8n_s32n_tensor_op_s32_perfsim.256x128x128_64x64x32",
"SM80_Device_Gemm_s8t_s8n_s32n_tensor_op_s32_perfsim.128x128x128_64x64x32"
]

# Update this to run a different set of conv tests
conv_binary = "/build/test/unit/conv/device/perfsim/cutlass_test_perf_conv_device_perfsim"
conv_testlist = [
 "SM80_Device_Precomputed_ImplicitGemm_Dgrad.NHWC_1x64x64x4096_KRSC_256x3x3x4096_tile_128x128_64x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Dgrad.NHWC_1x64x64x4096_KRSC_256x3x3x4096_tile_128x128_32x4stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Dgrad.NHWC_1x64x64x4096_KRSC_256x3x3x4096_tile_256x128_64x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Dgrad.NHWC_1x64x64x4096_KRSC_256x3x3x4096_tile_256x128_32x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_256x128_64x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_256x128_32x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_256x64_64x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_256x64_32x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_128x256_32x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_128x128_32x4stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_128x128_64x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_128x64_64x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_128x64_32x5stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_64x128_32x5stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_64x64_64x4stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Fprop.NHWC_1x64x64x256_KRSC_4096x3x3x256_tile_64x64_32x5stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Wgrad.NHWC_1x32x64x512_KRSC_4096x3x3x512_tile_128x128_64x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Wgrad.NHWC_1x32x64x512_KRSC_4096x3x3x512_tile_128x128_32x4stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Wgrad.NHWC_1x32x64x512_KRSC_4096x3x3x512_tile_256x64_32x3stage_warp_64x64x16",
 "SM80_Device_Precomputed_ImplicitGemm_Wgrad.NHWC_1x32x64x512_KRSC_4096x3x3x512_tile_256x128_32x3stage_warp_64x64x16"
]

##########################################################
##
##  DO NOT EDIT BELOW THIS POINT OR EDIT AT YOUR OWN RISK
##
###########################################################

def generate_json(binary_list, testlists_, filename):
    global cudacxx, run_dir, arch, project, flow_config_file, flow_perfsim_path, ld_library_path, cutlass_pull_branch
    test_config = {
                        "CUDACXX"   : cudacxx,
                        "RUN_DIR"   : run_dir,
                        "ARCH"      : arch,
                        "PROJECT"   : project,
                        "FLOW_CONFIG_FILE"  : flow_config_file,
                        "FLOW_PERFSIM_PATH" : flow_perfsim_path,
                        "LD_LIBRARY_PATH"   : ld_library_path,
                        "CUTLASS_PULL_BRANCH" : cutlass_pull_branch,
                        "WORKLOAD"  : []
                  }

    binary_index = 0
    for binary in binary_list:

        binary_config = {
                        "BINARY" : binary,
                        "TESTS"  : []
                        }

        testlist = testlists_[binary_index]    
        binary_index += 1
    
        for t in testlist:
            binary_config["TESTS"].append({
                                "NAME" : t,
                                "ARGS" : "--gtest_filter=\"{0}\"".format(t)
                                })

        test_config["WORKLOAD"].append(binary_config)

    fp = open(filename, "w")
    json.dump(test_config, fp, indent=1)


def config_generator(flag = "all"):
    global binary_list, test_list

    if flag == "all":
        # If they are empty, fill with gemm + conv
        if binary_list[0] == None and test_list[0] == None:
            binary_list = [gemm_binary, conv_binary]
            test_list = [gemm_testlist, conv_testlist]
        filename = "config.json"

    elif flag == "gemm":
        binary_list = [gemm_binary]
        test_list = [gemm_testlist]
        filename = "gemm_config.json"

    elif flag == "conv":
        binary_list = [conv_binary]
        test_list = [conv_testlist]
        filename = "conv_config.json"
    
    else:
        print("ERROR : Invalid flag passed to json generator")
        exit(1)

    generate_json(binary_list, test_list, filename)

if __name__ == "__main__" :
    config_generator("all")
