from .cnexus_perfdb_v2 import Api, Cpu, Env, Gpu, System
from .cnexus_perfdb_v2.base import NVDATAFLOW_PREFIX_MAPPING
from .query import search_dlsim_records
from .models import Build, PerfRecord, Test, Testlist
from .utils import *

import argparse
import csv
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

api = Api(env=Env.DEV)

def process_test_info(args):
    logger.info("Collecting build and test info")

    with open("testSpec.json") as f:
        test_spec = json.load(f)

    build_common = test_spec["buildCommon"]

    if revision := getCudaRevision(env=args['env']):
        cuda_revision = f" (nvcc@{revision})"
    else:
        cuda_revision = ""

    git_info = getGitInfo()

    build = Build(
        git_branch=git_info["branch"],
        git_commit_id=git_info["commit_id"],
        git_commit_date=git_info["commit_date"],
        cuda=build_common["cuda"] + cuda_revision,
        cuda_toolkit_artifact=getCudaArtifact(build_common["cuda"]),
        pipeline_artifact_url=build_common["artifactUrl"],
        pipeline_build_suffix=build_common["buildSuffix"],
    )
    
    testlist = Testlist(
        name=args["network_type"],
    )

    test = Test(
        build=build,
        testlist=testlist,
        system=System.from_current(),
        platform=args["gpu"],
        cpu=Cpu.from_current(),
    )
    
    gpus = Gpu.from_current()
    if len(gpus) > 0:
        gpu = gpus[0]

    return test


def process_test_records(args, test, tier):
    logger.info("Processing test records")
    new_perf_records = []

    for workload_id, value_dict in args["perf_records"].items():
        # construct perf_record to be stored in perf database
        new_perf_record = PerfRecord(
            test_id=test.id,
            case_id=workload_id,
            tier=tier,
            input_params=value_dict.get("input_params"),
            runtime_params=value_dict.get("runtime_params"),
            metrics=value_dict.get("metrics"),
        )

        new_perf_records.append(new_perf_record)

    return new_perf_records


def post_process(args):
    if type(args) is argparse.Namespace:
        args = vars(args) # to dict

    global logger
    if "logger" in args:
        logger = args["logger"]

    test = process_test_info(args)

    perf_records_tier_0 = process_test_records(args=args, test=test, tier=0)
    # perf_records_tier_1 = process_test_records(args=args, test=test, tier=1)

    all_records = []
    all_records.extend(perf_records_tier_0)
    # all_records.extend(perf_records_tier_1)
    all_records.append(test)

    if len(all_records) > 1:
        api.push(all_records)
        api.create_archive(all_records, "{}/{}".format(args["output_path"], f"perf_results.archive"))
        logger.info(f"Successfully pushed {len(all_records)} records")
    else:
        logger.warning("No perf records, skip uploading to perf database")
