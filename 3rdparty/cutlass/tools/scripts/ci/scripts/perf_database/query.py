from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

client = Elasticsearch(hosts="https://gpuwa.nvidia.com/elasticsearch")
index = "df-compute_arch-cnexus_perfdb_v2-development-fast_kernels_perf_regression-{}-*"

def search_latest_test(branch, testlist):
    s = Search(using=client, index=index.format("test")) \
            .exclude("term", b_invalid=True) \
            .filter("term", build__s_git_branch=branch) \
            .filter("term", testlist__s_name=testlist) \
            .sort("-build.ts_git_commit_date", "-@timestamp") \
            .extra(size=1)

    for hit in s:
        return hit.to_dict()

def search_perf_records(test, tier):
    s = Search(using=client, index=index.format("perf_record")) \
            .exclude("term", b_invalid=True) \
            .filter("term", s_test_id=test["s_id"]) \
            .filter("term", l_tier=tier)
    
    records = []

    for hit in s.scan():
        records.append(hit.to_dict())

    return records

def search_dlsim_records(testlist, tier):
    s = Search(using=client, index=index.format("dlsim_record")) \
            .exclude("term", b_invalid=True) \
            .filter("term", testlist__s_name=testlist) \
            .filter("term", l_tier=tier)
    
    records = {}

    for hit in s.scan():
        record = hit.to_dict()
        record["meta"] = hit.meta.to_dict()
        records[record["s_case_id"]] = record
    
    return records

def search_dlsim_record(case_id):
    s = Search(using=client, index=index.format("dlsim_record")) \
            .exclude("term", b_invalid=True) \
            .filter("term", s_case_id=case_id)
    
    for hit in s:
        return hit.to_dict()


if __name__ == '__main__':
    test = search_latest_test("dev", "p1_cublas_hsh_0")
    records = search_perf_records(test)
    print(len(records))
    print(records[0])
