#!/home/utils/Python-3.6.1/bin/python3.6
import os, stat
import json
import sys
import subprocess
import datetime
import argparse
from generate_json import *


# STEPS INVOLVED IN PERFSIM RUNNER :
# 1. Generate the JSON version of the config {we have adopted json since it naturally maps to python dict}
# 2. If JSON already provided - parse it
# 3. Generate the YML file with the appropriate format
# 4. Generate a shell script to run on LSF
# 5. Launch script on LSF 

# If value not provided - these will be used (should not be necessary if json is generated using generate_json.py)
def set_defaults(config):
    default_config = {
                        "CUDACXX" : "/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/bin/nvcc",
                        "ARCH"    : 82,
                        "LD_LIBRARY_PATH" : "/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/lib64",
                        "CUTLASS_DEV"     : "ssh://git@gitlab-master.nvidia.com:12051/dlarch-fastkernels/cutlass.git",
                        "CUTLASS_PULL_BRANCH" : "dev",
                        # TODO : Update this path to path inside the localy created branch
                        "FLOW_CONFIG_FILE": "/home/scratch.prramani_gpu_4/cutlass_perfsim_runs/perfsim-runner/config.perfsim.yml",
                        "FLOW_PERFSIM_PATH" : "/home/scratch.svc_compute_arch/release/flow.perfsim/latest/flow_perfsim/flow.perfsim",
                        "RUN_DIR" : "/home/scratch.prramani_gpu_4/cutlass_perfsim_runs/"
                     }

    for key in default_config :
        if not key in config:
            config[key] = default_config[key]

    return config


## This assumes that the name of the file is cutlass_perfsim_config.json
def parse_config(filename):
    if os.path.isfile(filename):
        f = open(filename , "r")
        config = json.load(f)

        config = set_defaults(config)
        return config
    else:
        print("ERROR : Missing cutlass_perfsim_config.json")
        exit(1)


def generate_tcsh_cmds(config):

    ########################################################
    # Some pre-work in checking paths
    ########################################################
    assert os.path.exists("/home/utils/Python-3.6.1/bin"), "ERROR : Check Python Path"
    assert os.path.exists("/home/utils/modules-tcl/init/tcsh"), "ERROR : Check tcsh modules init path"
    assert os.path.exists("/home/utils/cmake-3.14.1/bin/cmake"), "ERROR : check cmake path"
    assert os.path.exists(config["CUDACXX"]), "ERROR : Invalid CUDACXX path : check config.json "
    assert os.path.exists(config["RUN_DIR"]), "ERROR : RUN_DIR does not exist - check your config.json / generate_json.py file"
    assert os.path.exists(config["FLOW_PERFSIM_PATH"]) , "ERROR : Check flow.perfsim path" 
    assert os.path.exists(config["LD_LIBRARY_PATH"]) , "ERROR : check LD_LIBRARY_PATH path" 
    assert os.path.exists(config["FLOW_CONFIG_FILE"]) , "ERROR : check FLOW_CONFIG_FILE path" 


    ########################################################
    # Repo creation commands (will be used later)
    # Create uniq repo, 
    # Create build, perfsim folders
    # Copy over the generated shell script and yaml file (safekeeping)
    ########################################################
    cutlass_repo_cmds  = "cd {0}                \n".format(config["RUN_DIR"])
    cutlass_repo_cmds += "git clone {0} {1}     \n".format(config["CUTLASS_DEV"], config["UNIQ_ID"])
    cutlass_repo_cmds += "cd {0}                \n".format(config["UNIQ_ID"])
    cutlass_repo_cmds += "mkdir build           \n"
    cutlass_repo_cmds += "mkdir perfsim         \n"
    cutlass_repo_cmds += "mv {} .               \n".format(os.path.join(config["RUN_DIR"], config["SHELL_SCRIPT"]))
    cutlass_repo_cmds += "mv {} .               \n".format(os.path.join(config["RUN_DIR"], config["WORKLOAD_YML"]))
    yaml_file_path = os.path.join(config["RUN_DIR"], config["UNIQ_ID"], config["WORKLOAD_YML"])

    
    ########################################################
    # Generation of the actual shell script
    # Set env variables 
    # Pull from dev/any branch if necessary, 
    # Cmake
    # Make
    # cd to perfsim directory
    # Launch perfsim with correct cmd
    ########################################################
    text  = "#!/usr/bin/env tcsh                                \n"
    text += "setenv PATH /home/utils/Python-3.6.1/bin\:$PATH    \n"
    text += "setenv CUDACXX {0}                                 \n".format(config["CUDACXX"])
    text += "source /home/utils/modules-tcl/init/tcsh           \n"
    text += "module load gcc/5.4.0                              \n"
    text += "module load doxygen/1.8.11                         \n"
    text += cutlass_repo_cmds                                   
    
    if config["CUTLASS_PULL_BRANCH"] != "dev":
        text += "git checkout {}    \n".format(config["CUTLASS_PULL_BRANCH"])
        if "CUTLASS_PULL_BRANCH_2" in config:
             text += "git checkout {}\n".format(config["CUTLASS_PULL_BRANCH_2"])
        text += "git checkout dev   \n"
        text += "git pull . {}      \n".format(config["CUTLASS_PULL_BRANCH"])
        if "CUTLASS_PULL_BRANCH_2" in config:
             text += "git pull . {}\n".format(config["CUTLASS_PULL_BRANCH_2"])
    
    text += "cd build\n"
    text += "/home/utils/cmake-3.14.1/bin/cmake .. -DCUTLASS_NVCC_ARCHS=" + str(config["ARCH"]) + " -DCUTLASS_ENABLE_F16C=0 > cmake_log.txt\n"

    if config["ONLY_CMAKE"] :
        return text

    if config["ONLY_CONV"]:
        text += "cd test/unit/conv/device/perfsim \n"
        text += "make -j4 > make_log.txt          \n"
        text += "cd ../../../../../../            \n"
    elif config["ONLY_GEMM"]:
        text += "cd test/unit/gemm/device/perfsim \n"
        text += "make -j4 > make_log.txt          \n"
        text += "cd ../../../../../../            \n"
    else:
        text += "cd test/unit/conv/device/perfsim \n"
        text += "make -j4 > make_log.txt          \n"
        text += "cd ../../../../../../            \n"
        text += "cd test/unit/gemm/device/perfsim \n"
        text += "make -j4 > make_log.txt          \n"
        text += "cd ../../../../../../            \n"
    
    if config["ONLY_BUILD"]:
        return text
    
    text += "cd perfsim  \n"


    if config["MORPH_ENABLE"]:
        text += "cp {0} . \n".format(config["FLOW_CONFIG_FILE"])
        text += "{0} run -cuda2ctlAmodel -par 8 -chip ga100 -workload {1} -enableMorph -config config.perfsim.yml  > flow_perfsim_log.txt   \n".format(config["FLOW_PERFSIM_PATH"], yaml_file_path)

    else:
        text += "cp {0} . \n".format(config["FLOW_CONFIG_FILE"])
        text += "{0} run -cuda2ctlAmodel -par 8 -chip ga100 -workload {1} -config config.perfsim.yml > flow_perfsim_log.txt\n".format(config["FLOW_PERFSIM_PATH"], yaml_file_path)
    
    return text

def generate_workload_yml(config):
    text  = config["PROJECT"]                               + ":\n"
    for tests in config["WORKLOAD"]:
        tests_binary  = config["RUN_DIR"] + "/" + config["UNIQ_ID"] + "/"+   tests["BINARY"]
        
        for test in tests["TESTS"]:
            text += "  " + test["NAME"]                     + ":\n"
            text += "    args: " + test["ARGS"]             + "\n"
            text += "    DumpControl: '(range=0:0)'"        + "\n"
            text += "    binary: " + tests_binary           + "\n"
            text += "    env:"                              + "\n"
            text += "      LD_LIBRARY_PATH: " + config["LD_LIBRARY_PATH"] + "\n"

    return text

def main():
    
    ########################################################
    # Steps involved :
    # 1. parse the arguments 
    # 2. Parse the JSON config file, if json not given use generation scripts to first create the json
    # 3. Generate Shell script
    # 4. Generate workload.yml file
    # 5. Launch the qsub command to run the shell script 
    ########################################################

    #### Argument Parser 
    parser = argparse.ArgumentParser(description='Process Arguments for Perfsim runner')
    parser.add_argument('--config'    , dest='config_filename'  , default=None , help="name of the config.json file")
    parser.add_argument('-o', '--output' , dest='output'        , default=None , help="Output path, sets RUN_DIR field")
    parser.add_argument("--only_conv" , action='store_true'     , default=False, help="Runs only conv tests")
    parser.add_argument("--only_gemm" , action='store_true'     , default=False, help="Runs only gemm tests")
    parser.add_argument("--only_cmake", action='store_true'     , default=False, help="Runs only cmake command, does not build")
    parser.add_argument("--only_build", action='store_true'     , default=False, help="Runs only make -j4 command, does not run perfsim")
    parser.add_argument("--name", dest='folder_name', default='', help="name of the Folder to be created")
    parser.add_argument("--morph_enable", action='store_true', default=False, help="Enable morph dump for 1 SM")
    parser.add_argument("--debug", action='store_true', default=False, help="Only dumps out the shell script, does not launch on LSF")
    args = parser.parse_args()


    #### Unique ID creation - this will be used throught to create folders, script names etc. 
    now = datetime.datetime.now()
    uniq_id = 'cutlass_' + args.folder_name + '_{}-{}-{}_{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    print("Creating Unique ID for this run : " + uniq_id)
    
    #### Generate (if necesary) and Parse the JSON config file 
    if args.config_filename == None:
        if args.only_conv:
            config_generator("conv")
            config = parse_config("conv_config.json")
        elif args.only_gemm:
            config_generator("gemm")
            config = parse_config("gemm_config.json")
        else:
            config_generator("all")
            config = parse_config("config.json")
    else:
        config = parse_config(args.config_filename)

    config["ONLY_CONV"] = args.only_conv
    config["ONLY_GEMM"] = args.only_gemm
    config["ONLY_CMAKE"]= args.only_cmake
    config["ONLY_BUILD"]= args.only_build
    config["MORPH_ENABLE"] = args.morph_enable
    config["UNIQ_ID"] = uniq_id
    config["SHELL_SCRIPT"] = "perfsim_cmds_{}.tcsh".format(uniq_id)
    config["WORKLOAD_YML"] = "workload_{}.yml".format(uniq_id)
    if args.output != None:
        config["RUN_DIR"] = args.output
        assert os.path.exists(config["RUN_DIR"]), "ERROR : RUN_DIR does not exist - check your cmdline"
        
    tcsh_file_path = os.path.join(config["RUN_DIR"] , config["SHELL_SCRIPT"])
    yml_file_path  = os.path.join(config["RUN_DIR"] , config["WORKLOAD_YML"])

    #### Generate and Dump the CSH script 
    tcsh_cmds = generate_tcsh_cmds(config)
    tcsh_file = open( tcsh_file_path, "w")
    os.chmod(tcsh_file_path, stat.S_IRWXU)
    tcsh_file.write(tcsh_cmds)

    #### Generate and Dump the Workload.yml file 
    yml_file_txt = generate_workload_yml(config)
    yml_file  = open( yml_file_path , "w")
    yml_file.write(yml_file_txt)


    #### Launch the tcsh script on LSF
    if not args.debug:
     subprocess.run(["qsub", "-q" , "o_build_cpu_16G_4H",  "tcsh " + tcsh_file_path, "-o", "qsub_log.txt"]) 
     

if __name__ == "__main__":
    main() 
