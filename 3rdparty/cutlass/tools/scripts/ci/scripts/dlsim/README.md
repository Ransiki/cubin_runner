# Overview
This module consists of a set of python, and shell scripts to help users to easily install, build and run dlsim on CUTLASS's pre-defined performance tests (e.g. perfsim testlist, details see `emit_kernel_listing.py`) in their desired working environment. 

This module also provide end-users the option to create/update dlsim record in cutlass performance database.

All scripts are designed to be runnable in a stand-alone fashion. Due to its modular nature, it is easy to extend its functionality to support more operation types in the future.

# Limitations
### System:
Linux is the only supported OS to run end-to-end **dlsim -> perf database** flow as of Aug/10/2023

This module has been tested to work with below platform:
- OS: Linux
- Distribution: Ubuntu 20.04.4 LTS
- Arch: x86_64
- Python: 3.8.2
- Shell: bash 4.2.46(2)-release
### Operation type
Only gemm is integrated to this module as of Aug/10/2023.
# Scripts and its functionalities
***_dlsim_install.sh_***
``` 
Download and build dlsim project

Usage:
$ ./dlsim_install.sh -w <abs_path_to_dlsim_project> -b (If specified, will rebuild dlsim from source)

Example:
$ chmod +x dlsim_install.sh

# Download dlsim source code to /tmp/workspace
$ ./dlsim_install.sh -w /tmp/workspace

# Download dlsim source code to /tmp/workspace, and build dlsim from source
$ ./dlsim_install.sh -w /tmp/workspace -b
```

***_dlsim_gemm_runner.sh_***
``` 
Run dlsim model with predefined testlist

Usage:
$ ./dlsim_gemm_runner.sh -v <abs_path_to_venv_where_dlsim_is_installed> -i <abs_path_to_infile_for_dlsim_run> [-p <rel_path_to_store_pi_results]

Example:
$ chmod +x dlsim_gemm_runner.py

# Run dlsim gemm explainer with testcases defined in /tmp/workspace/gemm.csv
$ ./dlsim_gemm_runner.py -v /tmp/workspace/dlsim/python/venv/bin/python3 -i /tmp/workspace/gemm.csv
```

***_dlsim_translator.py_***
```
Map cutlass_profiler command to equivalent dlsim cmd

Usage:
$ python3 -m dlsim.dlsim_translator --perf_list <perf_perfsim, perf_smart(not supported yet)> --input_file <abs_path_to_csv_testlist_with_profiler_cmds> --output_file <abs_path_to_generated_csv_with_dlsim_testlist> --cc <sm100> [--pi_path <relative_path> (if used, sets per-case pi result path for batch mode)]

Example:
$ cd cutlass/tools/cripts/ci/scripts

# Translate profiler commands to dlsim testlist
$ python3 -m dlsim.dlsim_translator --perf_list perf_perfsim --input_file /tmp/workspace/FK_perf_perfsim_testlist_SM100_cutlass3x_gemm.csv --output_file /tmp/workspace/gemm.csv
```

***_dlsim_runner.py_***
```
Run end-to-end flow, a complete flow is describe as below (some steps is optional)
1. User specify which perf testlist to run (e.g. perf_perfsim)
2. Script calls generator.py internally to generate profiler-based performance testlist
3. Script calls translate() to map profiler-based performance testlist to dlsim testlist
4. Script calls dlsim_gemm_runner.sh to run dlsim testlist
5. Script parse dlsim test results, and push to datbase (with --real_run flag, otherwise (dryrun) the script writes perf records to local file) 

Usage:
$ python3 -m dlsim.dlsim_runner --perf_list <perf_perfsim, perf_smart(not supported yet)> --workspace <abs_path_to_current_working_directory> --cleanup (if turning on, will cleanup everything in the workspace) --build_dlsim (if turning on, will build dlsim from source) --real_run (if turning on, will push/update perf records in database) --cc <sm100> --pi_path <path_relative_to_workspace> (if used, will generate dlsim phase view PI results in pi_path)

Example:
$ cd cutlass/tools/cripts/ci/scripts

# Local dryrun (write records to local filesystem)
$ python3 -m dlsim.dlsim_runner --perf_list perf_dlsim --workspace /tmp/workspace --cleanup --build_dlsim --cc sm100

# Actual run (push/update records in performance database)
$ python3 -m dlsim.dlsim_runner --perf_list perf_dlsim --workspace /tmp/workspace --cleanup --build_dlsim --real_run --cc sm100

# When using --pi_path, be sure to set your workspace to a valid scratch space for valid PI links
$ python3 -m dlsim.dlsim_runner --perf_list perf_dlsim --workspace /home/scratch.<your_scratch>/<dlsim_workspace_dir> --cleanup --build_dlsim --cc sm100 --pi_path PI_results
```
