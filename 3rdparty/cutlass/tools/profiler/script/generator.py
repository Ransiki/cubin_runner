# {$nv-internal-release file}
#
# Generate ctest for performance regression.

import os
import argparse
import tempfile
from pathlib import Path
from platform import architecture
from csv_utils import preprocess_testlist_csv, csv_reader_fn

SCRIPT_FILE = os.path.abspath(__file__)
DIRNAME = os.path.dirname(SCRIPT_FILE)

PERF_TESTLIST_LOOKUP = {
    "compiler_sm80_blas3": "FK_Compiler_perf_testlist_GA100_SM80_cutlass_blas3.csv",
    "compiler_sm80_conv": "FK_Compiler_perf_testlist_GA100_SM80_cutlass_conv.csv",
    "compiler_sm80_gemm": "FK_Compiler_perf_testlist_GA100_SM80_cutlass_gemm.csv"
}

CUTLASS_PROFILER_PARAMS_OPTIONS = [
    "--append=true",                            # If true, result is appended to possibly exisiting file.
    "--output=/tmp/test_cutlass_performance",   # Path to output csv file (<report>.<operation kind>.csv).
    "--junit-output=test_cutlass_performance"   # Path to junit output file for result reporting (<report>.<operation kind>.junit.xml)
]

def workspace_directory(args) -> tempfile.TemporaryDirectory:
    '''Create and return a temporary directory under args.workspace.
    
    To close and remove the directory and its contents after use,
    either use the result in a 'with' statement,
    or invoke the .close() method on it.
    '''
    if not hasattr(args, 'workspace') or args.workspace is None:
        tmpdir_base = Path.cwd()
    else:
        tmpdir_base = Path(args.workspace)

    if not tmpdir_base.exists() or not tmpdir_base.is_dir():
        errstr = ('You specified the workspace directory as '
            f'--workspace={args.workspace}, but that either '
            'does not exist or is not a directory.')
        raise FileNotFoundError(errstr)
    return tempfile.TemporaryDirectory(dir=tmpdir_base)

def generate_cmake(args, cutlass_profiler_params: list = CUTLASS_PROFILER_PARAMS_OPTIONS):

    with workspace_directory(args.workspace) as testlist_temp_dir:
        architecture_list = args.architecture.split(";")
        test_list = args.test_list.split(";")

        # the "actual name" for the csv testlists
        perf_testlist = []

        for arch in architecture_list:
            for test in test_list:
                test_csv = PERF_TESTLIST_LOOKUP[f"compiler_sm{arch}_{test}"]
                perf_testlist.append(test_csv)

        testlist_temp_dir_path = Path(testlist_temp_dir)
        preprocess_testlist_csv(perf_testlist, testlist_temp_dir_path)

        test_command_options = []
        test_command_options_name = []
        count = 0
        command_option_list_base_name = "CUTLASS_PROFILER_PERFORMANCE_COMMAND_OPTIONS"
        cmake_set = ""

        for test_csv in perf_testlist:
            for cmd in csv_reader_fn(testlist_temp_dir_path / test_csv, dict_reader = False):
                # write a string
                if cmd and cmd[0] == "cutlass_profiler":
                    for i in range(len(cutlass_profiler_params)):
                        if "junit" in cutlass_profiler_params[i]:
                            length = len(cutlass_profiler_params[i])
                            for index, char in enumerate(cutlass_profiler_params[i][::-1]):
                                if (not char.isdigit()) and char != "_":
                                    cutlass_profiler_params[i] = cutlass_profiler_params[i][:length - index]
                                    break
                            cutlass_profiler_params[i] += f"_{str(count)}"
                    command_option_list_name = command_option_list_base_name + f"_{str(count)}"
                    test_command_options.append("set("+ " ".join([command_option_list_name] + cmd[1:] + cutlass_profiler_params) + ")")
                    test_command_options_name.append(command_option_list_name)
                    count += 1
        
        test_command_options = "\n".join(test_command_options)
        test_command_options_name = "\n    ".join(test_command_options_name)

        add_executable_tests = f"""
    cutlass_add_executable_tests(
    test_performance cutlass_profiler
    DEPENDEES test_all
    TEST_COMMAND_OPTIONS
        {test_command_options_name}
    DISABLE_EXECUTABLE_INSTALL_RULE
    )
    """

        with open("performance.cmake", 'w') as cmake_file:
            cmake_file.write(test_command_options)
            cmake_file.write(add_executable_tests)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates CMake commands")
    parser.add_argument("--architecture", default="80",
        help="Comma-delimited compute architectures")
    parser.add_argument("--test-list", default="gemm;conv;blas3",
        help="Comma-delimited list of tests in csv format, under /compiler_testlists")
    parser.add_argument("--workspace", default=str(Path.cwd()),
        help=('Name of a directory (write permissions required) '
            'to use for temporary files.  The files '
            '(but not the directory) will be deleted after use.'))

    args = parser.parse_args()

    generate_cmake(args)
