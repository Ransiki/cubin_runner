# {$nv-internal-release file}
#
# Utility functions for csv related operation

import os
import csv
from pathlib import Path, PurePath

# For backwards compatibility, we retain use of os.path.abspath,
# because it normalizes the path, while Path.absolute() does not.
SCRIPT_FILE = Path(os.path.abspath(__file__))
DIRNAME = SCRIPT_FILE.parent

PATH_TO_PERF_TESTLISTS = DIRNAME.joinpath("../../../compiler_testlists/")

def csv_reader_fn(file_name, dict_reader = True, delimiter = " "):
    fn = csv.DictReader if dict_reader else csv.reader

    with open(file_name, 'r') as csv_file:
        reader = fn(csv_file, delimiter = delimiter)
        for row in reader:
            row = list(filter(None, row)) # filter out empty string
            yield row

def preprocess_testlist_csv(testcsv_list: list, testlist_write_base_path: Path, logging: bool = False):
    def preprocess_helper(testlist_file_name):
        testlist_file_read_path = PATH_TO_PERF_TESTLISTS / testlist_file_name
        testlist_file_write_path = testlist_write_base_path / testlist_file_name

        csv_reader = csv_reader_fn(testlist_file_read_path, dict_reader = False)
        with open(testlist_file_write_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            for row in csv_reader:
                new_row = []
                write = False
                for item in row:
                    item = item.replace('\t', " ")
                    if "cutlass_profiler" in item: write = True
                    if not item.split()[0].isdigit(): new_row.append(item)
                new_row = " ".join(new_row)
                if write: csv_writer.writerow([new_row])
                elif logging: print(f"Testlist preprocess: skipping {new_row}")

    for testlist_file_name in testcsv_list:
        preprocess_helper(testlist_file_name)
