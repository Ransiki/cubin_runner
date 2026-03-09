import json
from datetime import datetime

from .cnexus_perfdb_v2 import Record, System, Gpu, Cpu, Optional
from .cnexus_perfdb_v2.base import Base, FlatDict

client = "fast_kernels_perf_regression"

class Build(Base):
    git_branch: str
    git_commit_id: str
    git_commit_date: datetime
    cuda: str
    cuda_toolkit_artifact: Optional[str]
    pipeline_artifact_url: str
    pipeline_build_suffix: str
    project="cutlass"

class Testlist(Base):
    name: str
    metadata: FlatDict={}

class Test(Record):
    _type='test'
    client=client
    timestamp=datetime.now()
    build: Build
    testlist: Testlist
    system: System
    platform: str
    gpu: Optional[Gpu]
    cpu: Cpu
    perf_scores: FlatDict={}

class PerfRecord(Record):
    _type='perf_record'
    client=client
    timestamp=datetime.now()
    test_id: str
    case_id: str 
    tier: int
    input_params: FlatDict={}
    runtime_params: FlatDict={}
    metrics: FlatDict={}

class DLSimRecord(PerfRecord):
    _type='dlsim_record'
    client=client
    test_id: Optional[str]
    testlist: Testlist
