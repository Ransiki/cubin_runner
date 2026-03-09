# translate cutlass profile cmdline to dlsim gemm-explainer

# External import
import csv
import argparse
import re
import os
from objdict import ObjDict

# Internal import
from . import PerfList
from run_presilicon_performance_tests import PERF_DB_NETWORK_PERFSIM_NAME, SM_TO_GPU_MAP, generate_workload_case_name
from helper.utils.perfsim_singlestep_html_parser import PerfsimHTMLParser


def create_cutlass_profiler_parser():
    parser = argparse.ArgumentParser(prog='cutlass_profiler command parser for DLSim')
    parser.add_argument('--m', type=int)
    parser.add_argument('--n', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--beta', type=int, default=0)
    parser.add_argument('--batch_count', type=int, default=1)
    parser.add_argument('--kernels')

    return parser


def decode_cutlass_kernel_name(kernel_name, perf_list):
    if perf_list == "perf_perfsim":
        data_type = [
            'f16_f16_f32_f16_f16',
            'gemm_f32_f32_f32_f32_f32_',
            'e4m3_e4m3_f32_f32_f32',
            's8_s8_s32_s8_s8',

            # void C kernels
            'f16_f16_f32_void_f16',
            'gemm_f32_f32_f32_void_f32_',
            'e4m3_e4m3_f32_void_f32',
            's8_s8_s32_void_s8',

            'ue8m0x(?:e2m1|e2m3)_ue8m0x(?:e2m1|e2m3)_f32_(?:void|f32|f16)_(?:e5m2|ue8m0xe2m1)',
        ]

        cgasize = ['4x4x1']
        layouts = ['tnn', 'ntt', 'ntn', 'tnt']
        tile_shape = ['(64|128|256|512)x\d+x\d+']
        instruction_shape = [
            # .2CTA
            ['128x128', '128x256', '256x128', '256x256']
        ]

        # regex list must be in kernel procedural name order
        filter_regex_2sm = ".*(" + ").*(".join([ "|".join(x) for x in [instruction_shape[0], data_type, tile_shape, cgasize, layouts]]) + ").*align[0-9]+_(.*)2sm.*"
        kernel_filter = f"({filter_regex_2sm})"
        mn_filter = f"((\d+)x(\d+))"
        mnk_filter = f"((\d+)x(\d+)x(\d+))"
        obj=re.match(kernel_filter, kernel_name, re.M)
        assert obj is not None, "Kernel does not match filter pattern"

        datatype = obj.group(3)
        tile_per_cga = obj.group(4)
        cga_string = obj.group(5)
        narrow_op = obj.group(8)
        cga_re = re.match(mnk_filter, cga_string)
        cga_tile = (4,4,1) # default value
        if cga_re:
            cga_tile = (int(cga_re.group(2)), int(cga_re.group(3)), int(cga_re.group(4)))

        # get tile size per CTA
        cta_re = re.match(mnk_filter, tile_per_cga)
        if cta_re:
            cga_tile_m, cga_tile_n,_ = cga_tile
            cta_tile = (int(cta_re.group(2)) // cga_tile_m, int(cta_re.group(3)) // cga_tile_n, int(cta_re.group(4)))

        # is sparse kernel
        is_sparse = "sptensorop" in kernel_name
        is_block_scaled = "block_scaled" in kernel_name


        vec_sz = "32"
        if is_block_scaled:
            op_vec = re.match('(q|o)_(vs(32|16))?', narrow_op)
            op = op_vec.group(1)
            vec = op_vec.group(3)

            if vec is not None:
                vec_sz = vec

        return (cta_tile, datatype, cga_tile, vec_sz, is_sparse, is_block_scaled)
    
    if perf_list == "perf_smart":
        pass
    
def get_dlsim_op(is_sparse, is_block_scaled, datatype):
    assert not (is_sparse and is_block_scaled)

    if is_block_scaled:
        B_type = datatype.split('_')[1].split('x')[1]
        
        # dlsim only supports e3m2, e2m1
        B_type = "e3m2" if B_type == "e2m3" else B_type
        B_type = "e2m1" if B_type == "e0m3" else B_type

        return B_type + ".fp32"


    dense_data_type_table = {
        'f16_f16_f32_f16_f16' : "hmma.fp32",
        'gemm_f32_f32_f32_f32_f32_' : "e8m10",
        'e4m3_e4m3_f32_f32_f32' :"e4m3.fp32",
        's8_s8_s32_s8_s8': "imma",

        # void C kernels
        'f16_f16_f32_void_f16': "hmma.fp32",
        'gemm_f32_f32_f32_void_f32_' : "e8m10",
        'e4m3_e4m3_f32_void_f32':"e4m3.fp32",
        's8_s8_s32_void_s8':"imma",
    }
    sparse_data_type_table = {
        'f16_f16_f32_f16_f16' : "hmma.fp32.sp",
        'gemm_f32_f32_f32_f32_f32_' : "e8m10.sp",
        'e4m3_e4m3_f32_f32_f32' :"e4m3.fp32.sp",
        's8_s8_s32_s8_s8': "imma.sp",

        # void C kernels
        'f16_f16_f32_void_f16': "hmma.fp32.sp",
        'gemm_f32_f32_f32_void_f32_' : "e8m10.sp",
        'e4m3_e4m3_f32_void_f32':"e4m3.fp32.sp",
        's8_s8_s32_void_s8':"imma.sp",
    }
    if is_sparse:
        return sparse_data_type_table[datatype]
    else:
        return dense_data_type_table[datatype]


def get_dlsim_gemm_precision_dict(is_block_scaled, datatype):
    gmem_type_table = {
        'e2m1' : "FP4",
        'e0m3' : "FP4",
        'e3m2' : "FP6",
        'e2m3' : "FP6",
        'e5m2' : "FP8"
    }

    if is_block_scaled:
        A_type  = datatype.split('_')[0].split('x')[1]
        B_type  = datatype.split('_')[1].split('x')[1]

        D_type = datatype.split('_')[-1]
        D_type = D_type if 'x' not in D_type else D_type.split('x')[1]

        gmem_in_A = gmem_type_table[A_type]
        gmem_in_B = gmem_type_table[B_type]

        gmem_out =  gmem_type_table[D_type]

        gemm_precision_dict = f"precision_gmem_in_A:{gmem_in_A};precision_gmem_in_B:{gmem_in_B};precision_gmem_out:{gmem_out}"
    else:
        gemm_precision_dict = None if datatype.split('_')[-1] != 'f32' else 'precision_gmem_out:FP32'


    return gemm_precision_dict

def convert_to_dlsim(batch, m, n, k, beta, kernel_name, perf_list, cc, pi_path):
    if perf_list == "perf_smart":
        return

    run_tag = PerfsimHTMLParser.generate_workload_id_for_perf_result(
        app_name= SM_TO_GPU_MAP[cc]["architecture"],
        workload_case_name=generate_workload_case_name(kernel=kernel_name, m=m, n=n, k=k)
    )

    cta_tile, datatype, cga_tile, vec_sz_, is_sparse, is_block_scaled = decode_cutlass_kernel_name(kernel_name, perf_list)
    run_tag_ = run_tag
    b_ = batch
    m_ = m
    n_ = n
    k_ = k
    beta_ = beta
    tiles_ = "x".join(str(x) for x in cta_tile)
    warp_tiles_ = "32x64"  # default value
    cga_tiles_ = "x".join(str(x) for x in cga_tile[0:2]) 
    op_ = get_dlsim_op(is_sparse, is_block_scaled, datatype)
    mode_ = "inference"
    task_ = "fprop"
    splitk_type_ = "nosplit"
    split_factors_ = 1

    gemm_precision_dict_ = get_dlsim_gemm_precision_dict(is_block_scaled, datatype)


    path_ = f"{pi_path}/{run_tag_}" if pi_path is not None else ""

    conf_mods_ = f'exec::scale_vector_sz::{vec_sz_}';

    return [run_tag_, b_, m_, n_, k_, beta, tiles_, warp_tiles_, cga_tiles_, op_, mode_, task_, splitk_type_, split_factors_, gemm_precision_dict_, path_, conf_mods_]


def translate(dlsim_translator_args):
    perf_list = dlsim_translator_args.perf_list
    input_file = dlsim_translator_args.input_file
    output_file = dlsim_translator_args.output_file
    cc = dlsim_translator_args.cc
    pi_path = dlsim_translator_args.pi_path

    print(f"Starting to translate profiler cmds to dlsim cmds for {perf_list} \ninput_file: {input_file} \noutput_file: {output_file} \ncc: {cc}\npi_path: {pi_path}")

    assert os.path.exists(input_file)
    os.remove(output_file) if os.path.exists(output_file) else None

    parser = create_cutlass_profiler_parser()
    columns = ["run_tag", "b", "m", "n", "k", "beta", "tiles", "warp_tiles", "cga_tiles", "op", "mode", "task", "splitk_type", "split_factors", "gemm_precision_dict", "pi_report_path", "extra_conf_mods"]

    with open(input_file, newline='') as in_file, open(output_file, "w", newline='') as out_file:
        csvreader = csv.reader(in_file, delimiter=',', quotechar='|')
        csvwriter = csv.writer(out_file)

        csvwriter.writerow(columns)

        next(csvreader)
        for row in csvreader:
            args, _ = parser.parse_known_args(row[0].split())
            line = convert_to_dlsim(args.batch_count, args.m, args.n, args.k, args.beta, args.kernels, perf_list, cc, pi_path)
            csvwriter.writerow(line)

    print(f"Write dlsim in-file to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DLSim translator input/output csv parser')
    parser.add_argument("--cc", default="sm100", help="target GPU's cc")
    parser.add_argument('--perf_list', default="perf_perfsim", type=PerfList, choices=list(PerfList), help="perf_perfsim or perf_smart")
    parser.add_argument('--input_file', help='path to input csv file, containing rows of cutlass_profiler cmd')
    parser.add_argument('--output_file', help='path to output csv, containing rows of dlsim cmd equavalent to cutlass_profiler cmd pre-defined input_file')
    parser.add_argument('--pi_path', default=None, help='path to dump dlsim PI report data')
    args = parser.parse_args()

    dlsim_translator_args = ObjDict({
        "perf_list": args.perf_list,
        "input_file": args.input_file,
        "output_file": args.output_file,
        "cc": args.cc,
        "pi_path": args.pi_path
    })

    translate(dlsim_translator_args)
