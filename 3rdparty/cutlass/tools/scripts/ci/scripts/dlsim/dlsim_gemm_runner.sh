#! /usr/bin/env bash
set -eo pipefail

print_usage() {
    printf "Usage: bash dlsim_runner.sh -v /tmp/workspace/dlsim/python/venv/bin/python3 -i /tmp/workspace/gemm.csv"
}

while getopts 'v:i:p:' flag; do
    case "${flag}" in
        v) PYTHON3="${OPTARG}" ;;
        i) INFILE="${OPTARG}" ;;
        p) PI_PATH="${OPTARG}" ;;
        *) print_usage
           exit 1 ;;
    esac
done

if [ -n "$PI_PATH" ]; then
	VISUAL="--enable_visual --pi_report_path=${PI_PATH} "
fi

echo "Running dlsim for gemm test cases"

# Invoke dlsim run
DLSIM_CMD="$PYTHON3 -m lwdlm.tools.gemm_explainer \
--device=GB100@1800/3600 \
--methodology=blackwell.conservative \
--device=GB100 \
--gpu_mods='{\"GB100\":[\"GB100\", {\"dram_clk\":3600,\"dram_interface\":\"HBM3\",\"gpu_clk\":1800,\"ltc_gpc_clkratio\":1.03333333333333,\"dram_latency\":1088,\"skyline\":\"10/10/10/10/10/10/10/10/1x0\",\"lrc_factor\":1.0}]}' \
--math_util 1 \
--l2_util 0.85 \
--dram_util 0.95 \
--cta_raster_order best-swizzle \
--conf_mods='exec::stages_model::true__exec::bw_based_empirical_math_deration::false__exec::empirical_math_deration::false__exec::cluster_mma_opts::{\"sm_org\":[\"2x1\",\"1x1\"]}__exec::tile_oob_model_level::quant_mnk__exec::swap_tile_dim::{\"sparse\":\"enable\",\"dense\":\"disable\"}__exec::dynamic_latency_model::{\"method\":\"piecewise_linear\"}' \
--batch_mode \
${VISUAL} \
--in_file $INFILE"

# Python caller process needs the piped stdout to parse dlsim cmd
echo "dlsim cmd: $DLSIM_CMD"

# Run dlsim command
eval "$DLSIM_CMD"
