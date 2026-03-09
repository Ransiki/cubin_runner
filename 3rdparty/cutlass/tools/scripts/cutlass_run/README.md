### Files
- cutlass_run.py: runs cutlass_profiler 
- cudnn_default.layer: contains CUDNN use case we care to do performance analysis
- cudnn_default.label: contains labels used in corrosponding layer
The original layer and label files are present in cudnn repo in P4. `//sw/gpgpu/MachineLearning/cudnn/scripts`

### Running layer files

#### Run TopN cublas  layers with cutlass_profiler

```python cutlass_run.py -binpath $build/tools/profiler -layer_file ./layers/cudnn_default.layer -label_file ./labels/cudnn_default.label -whitelist_layer_name "TestLayer" -config FAST```


#### Run Resnet50 Fprop layers with cutlass_profiler

```python cutlass_run.py -binpath $build/tools/profiler -layer_file ./layers/cudnn_default.layer -label_file ./labels/cudnn_default.label -whitelist_layer_name "ResNet50_v1" -global_flags " R:conv * n:16" -config FAST```

#### Run Resnet50 Dgrad layers with cutlass_profiler

```python cutlass_run.py -binpath $build/tools/profiler -layer_file ./layers/cudnn_default.layer -label_file ./labels/cudnn_default.label -whitelist_layer_name "ResNet50_v1" -global_flags " R:dgrad * n:16" -config FAST```

#### Run Resnet50 Wgard layers with cutlass_profiler

```python cutlass_run.py -binpath $build/tools/profiler -layer_file ./layers/cudnn_default.layer -label_file ./labels/cudnn_default.label -whitelist_layer_name "ResNet50_v1" -global_flags " R:wgrad * n:16" -config FAST```


#### Layer Flags to cutlass_profiler command generator
See/Edit: generate_cutlass_profiler_flags


### Extracting information from the runs

Add a regex to the LogExtractor here: https://gitlab-master.nvidia.com/dlarch-fastkernels/cutlass/blob/feature/2.x/scripts/tools/scripts/cutlass_run/helpers/cutlass_interface.py#L196

Call python spreadsheet giving the generated logs and field you want to extract in the csv: For example, "python spreadsheet.py -log_paths ~/_ga100_logs/cutlass_Renset50_fprop.log -extract time".

### Locking GPU clocks and pass correct locked clocks to cutlass_perf_test
- Best way to lock locks: https://confluence.nvidia.com/display/DL/How+to+lock+GPU+clock

#### Additional Info on Locking the Clocks
- User should run the following command before setting clocks of gpus `sudo nvidia-smi -acp UNRESTRICTED`. This command makes all users control gpu clocks without requiring `sudo` access to subsequent `nvidia-smi` commands
- query possible clocks, use `nvidia-smi -q -d SUPPORTED_CLOCKS -i <DEVICE_ID>`
- set desired SM and MEM clocks, use `nvidia-smi -ac memory_clk,graphic_clk`. On GV100 set `nvidia-smi -ac 850,1500`
- consider setting persistent mode `nvidia-smi -i <DEVICE_ID> -pm 1`
- run `nvidia-smi dmon -i <DEVICE_ID>` in a separated terminal, it will show you the current temperature, memory clock and graphics clock every second
- `--clock=<MHz>`records the clocks for cutlass_perf_test to compute correct utilization

#### Things to do before running `cutlass_run.py`
This section explains how to setup environment to run layer files with automatic clock settings for given devices

1. Make sure that pandas package is installed for python2.7. It can be installed by running `pip2 install --user pandas`
2. The cutlass_run script sets GPU clocks to desired values for a GPU automatically. So, it looks for file name `power_gpu.csv` present in `helpers/` directory. Make sure to add *Name of GPU*, *SM architecture number*, *GPU core clock* and *GPU memory clock* to a new line in the file. Make sure to follow CSV format conventions while adding data. The script looks up power profiles through *Name of GPU* so, make sure to use the same name what CUDA driver provides.
3. Once `power_gpu.csv` file is in place, user have to set GPU power profiles under *UNRESTRICTED* and *persistence* mode. These can be done by running `sudo nvidia-smi -acp UNRESTRICTED` and `sudo nvidia-smi -pm 1 -i <device id>`. If the command is not run, the script exits printing out error output from nvidia-smi. The feature of setting automatic gpu clocks can be bypassed using `--permit-unlocked-clocks` which disables setting of GPU clocks. This feature is important as user may forget to set gpu clocks before running or user can share these profiles with different users to reproduce performance results
4. If you have multiple gpus on your machine, you can add `-d <cuda device id>` flag to `cutlass_run.py`. Make sure that nvidia-smi device ids are same as cuda device ids

#### How to run `cutlass_run.py`
By runninng `python2 cutlass_run.py`, emits help
An example of running the script for convolution is ```python cutlass_run.py -binpath $build/tools/profiler -layer_file ./layers/cudnn_default.layer -label_file ./labels/cudnn_default.label -whitelist_layer_name "ResNet50_v1" -global_flags " R:conv * n:16" -config FAST```
For Gemm, ```python cutlass_run.py -binpath $build/tools/profiler -layer_file ./layers/cublas_perf.layer -label_File ./labels/cublas_perf.label```

#### What happens after running `cutlass_run.py`
1. After all runs, script dumps results to a csv file. The directory in which they are put can be controlled using the flag `--csv_path <path>`. The script tries to create that directory by default. The default directory which the script dumps is `./logs_csv/`
2. The name of logs dumped into directory follow the following naming convention `<date>_<time>_<layer-file>_<label-file>_<conv/gemm>_<all/no_output/bad_return>.csv`


### `perf_analysis.py`
This script takes csv generated by `cutlass_run.py` and generate visualization html/js files. For visualization we generate 3 types of graphs.
1. Absolute performance bar graph
2. Max performance bar graph compared to cuDNN or cuBLAS
3. Relative performance S curve

#### Things to do before running `perf_analysis.py`
1. Make sure that pandas package is installed for python2.7. It can be installed by running `pip2 install --user pandas`
2. The name of the csv file should contain either `conv` or `gemm` in its name.
3. html files present in ./assets directory

#### How to run `perf_analysis.py`
By running `python2 perf_analysis.py` emits help.
An example, ```python2 perf_analysis.py --csv_file ./logs_csv/<csv file name>```

#### What happens after running `perf_analysis.py`
1. The `./assets/data` directory gets filled with javascript and html files. The `perf_analysis.py` script generates 3 html files and 3 javascript files. What are they?
a. The 3 javascript files contain js function returning csv formatted data for absolute performance in gflops, maximum performance compared against cudnn/cublas kernels, relative performance against cudnn/cublas kernels
b. The 3 html files contain tables to map problem ids (x-axis) from the javascript graphs to relevant data from benchmarking run. As there are 3 javascript files generated, 3 html tables are generated one for each

The trend of file names are:
1. xxx_data_perf.js, xxx_table_xxx_perf.html, have graphs and data about absolute performance of CUTLASS
2. xxx_data_max_perf.js, xxx_table_xxx_max_perf.html, have graphs and data about maximum performance of CUTLASS along with best performing cuBLAS or cuDNN kernels
3. xxx_data_rel_perf.js, xxx_table_xxx_rel_perf.html, have graphs and data about relative performance of CUTLASS against cuBLAS or cuDNN kernels
