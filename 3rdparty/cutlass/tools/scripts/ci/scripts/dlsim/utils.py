# External import
import os
import sys
import json
from pathlib import Path


# Internal import
sys.path.append(str(Path(os.path.realpath(__file__)).parent.parent))
from perf_database.cnexus_perfdb_v2 import Api, Env
from perf_database.models import DLSimRecord, Testlist
from perf_database.query import search_dlsim_records

api = Api(env=Env.DEV)

def compute_dlsim_mainloop_cycles(result):
    GPC_CLK = 1800
    mainloop_cycles = float(result["Cycles"]) - float(result["launch_cycles"]) - (float(result["DRAM Latency"]) * GPC_CLK * 1000) - float(json.loads(result["CTA Breakdown"].replace("'", "\""))['Epilog Cycles'])

    return mainloop_cycles

def correlate_dlsim(workspace, dlsim_result, testlist, real_run):
    existing_records = 0

    # Needed for dryrun
    dry_run_needs_update = []
    dry_run_needs_create = []
    dry_run_update_records = os.path.join(workspace, "dryrun_update_records.json")
    dry_run_new_records = os.path.join(workspace, "dryrun_new_records.json")

    # Get all dlsim records (testlist, tier 0) from perf database (ES)
    all_dlsim_records = search_dlsim_records(testlist, 0)


    for case_id, result in dlsim_result.items():
        result["Mainloop Cycles"] = compute_dlsim_mainloop_cycles(result)

        if case_id not in all_dlsim_records:
            dlsim_record = DLSimRecord(
                testlist=Testlist(name=testlist),
                case_id=case_id,
                tier=0,
                input_params={
                    "workload_id": case_id   # FIXME: Need better handle to fetch input_params from perf_record
                },
                runtime_params={
                    "cmd": result["dlsim_cmd"]
                },
                metrics=result
            )
            if real_run:
                api.queue(dlsim_record)
            else:
                dry_run_needs_create.append(json.dumps(dlsim_record, indent=4, sort_keys=True, default=str))
        else:
            dlsim_record = all_dlsim_records[case_id]
            dlsim_record_to_update = {
                "_index": dlsim_record["meta"]["index"],
                "_id": dlsim_record["meta"]["id"],
                "_source": {
                    "ts_created": dlsim_record["ts_created"],
                }
            }

            if real_run:
                api._update_record(
                    dlsim_record_to_update,
                    update_callback=(lambda d: {**d, "flat_metrics": {"Cycles": result["Cycles"], "Mainloop Cycles": result["Mainloop Cycles"], "PI Report Link": result["PI Report Link"]}, "flat_runtime_params": {"cmd": result["dlsim_cmd"]}})
                )
            else:
                dry_run_needs_update.append(dlsim_record_to_update)

            existing_records += 1

    if real_run:
        new_records = len(api._queued)
        if new_records > 0:
            api.push_queued()
        
        print(f"Successfully pushed {new_records} new dlsim records")
        print(f"Successfully updated {existing_records} existing dlsim records")
    else:
        with open(dry_run_new_records, "w") as f1, open(dry_run_update_records, "w") as f2:
            _1 = json.dumps(dry_run_needs_create)
            _2 = json.dumps(dry_run_needs_update)
            f1.write(_1)
            f2.write(_2)
        
        print(f"Check file {dry_run_new_records} for dry run results on newly created dlsim records, number={len(dry_run_needs_create)}")
        print(f"Check file {dry_run_update_records} for dry run results on updated dlsim records, number={len(dry_run_needs_update)}")
