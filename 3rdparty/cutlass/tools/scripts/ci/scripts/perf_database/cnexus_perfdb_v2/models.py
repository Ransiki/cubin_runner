import re
import uuid
import copy
import datetime
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from pydantic import validator
from pydantic.class_validators import root_validator

from .base import Base, Text, Extra, FlatDict
from .enums import MetricType


def _client_validator(v):
    if not re.compile(r"^[a-zA-Z0-9_.]+$").match(v):
        raise ValueError("Client may only contain these characters: [a-zA-Z_.]")
    return v


def _measurement_validator(cls, v):
    return {"value": v} if isinstance(v, (int, float, str)) else v


def _random_uuid(cls, v):
    # Set a unique, random value if none is specified.
    return v or str(uuid.uuid4())


def _component_efficiency(
    instance: "PerformanceResult", component: str
) -> Optional["Measurement"]:
    component: Optional[Measurement] = getattr(instance, f"{component}_power")
    component_mean_power: float = component.mean if component else 0.0
    throughput: Optional[Measurement] = instance.throughput

    if component_mean_power > 0.0 and throughput is not None:
        return Measurement(
            **{
                key: (
                    None
                    if getattr(throughput, key) is None
                    else getattr(throughput, key) / component_mean_power
                )
                for key in ["value", "min", "max", "mean", "target"]
            },
            percentiles={
                key: throughput.percentiles[key] / component_mean_power
                for key in throughput.percentiles.keys()
            },
            extra={
                key: throughput.extra[key] / component_mean_power
                for key in throughput.extra.keys()
                if isinstance(throughput.extra[key], (float, int))
            },
        )


class Measurement(Base):
    """
    A group of values for a single measurement (eg. latency), which includes max, min, mean, percentiles, etc.

    The units:
    - time: seconds
    - power: watts
    - clock speed: MHz
    - memory: MB
    - throughput: queries/second (q/s)
    """

    _aliases = {"total": "value"}

    # Public API
    value: Optional[float]  #: Default value
    min: Optional[float]  #:
    max: Optional[float]  #:
    mean: Optional[float]  #:
    target: Optional[float]  #:
    percentiles: Dict[
        int, float
    ] = {}  #: Percentile values, eg. {99: 0.1, 95: 0.1, ...}

    extra: Extra = {}  #:

    def __imul__(self, other: Union[int, float, "Measurement"]):
        """
        For scaling a measurement by a float easily, useful when converting one
        unit to another. E.g. To convert a measurement object from minutes to
        seconds, do `measurement_instance *= 60`. Does not modify anything in
        the extra field.
        """
        multiplier: float
        if isinstance(other, Measurement) and other.value is not None:
            multiplier = float(other.value)
        elif isinstance(other, (int, float)):
            multiplier = float(other)
        else:
            raise TypeError(
                "Only able to multiply measurement "
                "by float or another Measurement instance "
                "(measurement.value must be present)"
            ) from None

        for attr in ["value", "min", "max", "mean", "target"]:
            value = getattr(self, attr)
            if value is not None:
                setattr(self, attr, value * multiplier)

        for key, value in self.percentiles.items():
            self.percentiles[key] = multiplier * value

    def __itruediv__(self, other: Union[int, float, "Measurement"]):
        """
        For scaling a measurement by a float easily, useful when converting one
        unit to another. E.g. To convert a measurement object from seconds to
        minutes, do `measurement_instance /= 60`. Does not modify anything in
        the extra field.
        """
        multiplier: float
        if isinstance(other, Measurement) and other.value is not None:
            multiplier = float(other.value)
        elif isinstance(other, (int, float)):
            multiplier = float(other)
        else:
            raise TypeError(
                "Only able to divide measurement "
                "by float or another Measurement instance "
                "(measurement.value must be present)"
            ) from None

        if multiplier == 0:
            raise ZeroDivisionError() from None

        self.__imul__(1 / multiplier)

    def __mul__(self, other: Union[int, float, "Measurement"]):
        """
        For scaling a measurement by a float easily, useful when converting one
        unit to another. E.g. To convert a measurement object from minutes to
        seconds, do `measurement_instance *= 60`. Does not modify anything in
        the extra field.
        """

        newCopyData = copy.deepcopy(self.dict())
        newCopy = Measurement(**newCopyData)
        newCopy *= other
        return newCopy

    def __truediv__(self, other: Union[int, float, "Measurement"]):
        """
        For scaling a measurement by a float easily, useful when converting one
        unit to another. E.g. To convert a measurement object from seconds to
        minutes, do `measurement_instance /= 60`. Does not modify anything in
        the extra field.
        """

        newCopyData = copy.deepcopy(self.dict())
        newCopy = Measurement(**newCopyData)
        newCopy /= other
        return newCopy

    @classmethod
    def create_measurement(
        cls,
        values: List[float],
        percentiles: Optional[List[float]] = [
            0,
            10,
            20,
            25,
            30,
            40,
            50,
            60,
            70,
            75,
            80,
            90,
            95,
            97,
            99,
            100,
        ],
    ) -> "Measurement":
        """
        Takes in a a list of values, and a list of percentiles to calculate,
        then returns a Measurement object containing percentiles (using the
        keys passed in), the mean, min and max. Does not set the value or target
        attributes of the `Measurement` object. Matches behavior of numpy
        percentile funciton.

        If an empty list is passed in, returns None.
        """
        if not values:
            return None

        values = sorted(values)

        # Calculate percentiles
        percentile_dict = {}
        for p in percentiles or []:
            if p < 0 or p > 100:
                continue  # skip over illegal percentiles
            if p == 100:
                percentile_dict[p] = values[-1]
                continue

            rank = (len(values) - 1) * (p / 100) + 1
            rank_int = int(rank)
            rank_float = rank - rank_int

            if rank_int >= len(values):
                rank_int = len(values) - 1
                rank_float = 0.0

            rank_int_value = values[rank_int]
            prev_rank_value = rank_int_value  # default

            if rank_int - 1 >= 0:
                prev_rank_value = values[rank_int - 1]

            percentile_dict[p] = round(
                sum([prev_rank_value, rank_float * (rank_int_value - prev_rank_value)]),
                2,
            )

        return Measurement(
            value=None,
            target=None,
            min=values[0],
            max=values[-1],
            mean=sum(values) / len(values),
            percentiles=percentile_dict,
        )


class Cpu(Base):
    """CPU information"""

    _aliases = {
        "cpu_count": "device_count",
        "product_name": "device_product_name",
        "shorthand_cpu_name": "short_name",
    }
    _duplicate_as = {Text: ["device_product_name", "short_name"]}

    @staticmethod
    def from_current():
        """Return a :py:class:`~Cpu` instance with data from the current machine."""
        from .utils.nrsu import get_cpu

        return get_cpu()

    device_product_name: Optional[str]  #:
    short_name: Optional[str]  #:

    architecture: Optional[str]
    """
    Name of ISA the CPU leverages.

    Example Value: `"x86_64"`
    """

    device_count: Optional[int]
    """
    Number of CPUs available on the baseboard.

    Example Value: `1`
    """

    cores_per_socket: Optional[int]  #:


class Gpu(Base):
    """GPU information"""

    _aliases = {
        "gpu_max_clock__MHz": "max_clock_rate",
        "mem_max_clock__MHz": "memory_max_clock_rate",
        "mem_total": "memory_total",
        "mem_total__MB": "memory_total",
        "mem_total__MiB": "memory_total",
        "power_limit__W": "power_limit",
        "vbios_version": "vbios",
    }
    _duplicate_as = {Text: ["device_product_name", "chip"]}

    @staticmethod
    def from_current():
        """Return a list of :py:class:`~Gpu` instances with data from the current machine."""
        from .utils.nrsu import get_gpus

        return get_gpus()

    # Public API

    device_product_name: str
    """
    A human friendly name of the card in use.

    Example Value: `"TITAN X"`
    """

    device_count: Optional[int]
    """
    Number of GPUs available.

    Example Value: `1`
    """

    chip: Optional[str]
    """
    The code for the chip in use.

    Example Value: `"gm200-a"` or `"tu102-a"`
    """

    compute_capability: Optional[float]
    """
    Compute capability refers to to a specific design member of an architecture
    generation. A full table of available values and their implications can be
    found in the `CUDA Documentation <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__feature-support-per-compute-capability_>`_

    For instance, 6.1 refers to pascal generation and 7.5 refers to turing
    generation.

    Example Value: `6.1` or `7.5`
    """

    uuid: Optional[str]
    """
    A unique identifier for this specific GPU.

    Example Value: `"GPU-2c78b86f-b978-bd0c-97f7-920484457184"`
    """

    pci_bus_id: Optional[str]
    """
    A nvidia specific representation of the bus id the card is using. Note the
    string is different than the return value of `lspci` command returns.
    The value from `nvidia-smi` must be used for this field.

    The format of the string is as follows:

    `<32 bit domain code>:<8 bit bus id>:<8 bit device id>.<function>`

    All codes are to be hexadecimal.

    Example Value: `"00000000:65:00.0"`
    """

    pci_device_id: Optional[str]
    """
    A 32 bit hexidecimal number representation of the device id.

    This value cannot be derrived from the `pci_bus_id`.

    Example Value: `"0x1FB910DE"`
    """

    power_limit: Optional[float]
    """
    Maximum Watts that can be fed to the card.
    """

    serial_number: Optional[str]
    """
    A string representation of a vendor issued number to the card.

    Example Value: `"0420215011116"`
    """

    memory_total: Optional[float]
    """
    Total frame buffer memory available on the card, in MiB.
    Does not have anything to do with BAR1 virtual memory.

    Note the difference between MB (base 1000) and MiB (base 1024).
    `nvidia-smi` reports MiB by default.

    Example Value: `40536`
    """

    multiprocessor_count: Optional[int]
    """
    Number of streaming multiprocessors (SM) available on the card.

    Example Value: `72`
    """

    vbios: Optional[str]
    """
    GPU Bios Version number.

    Example Value: `"92.00.35.00.00"`
    """

    index: Optional[int]
    """
    In multi GPU systems, value represents the nvml 0-based index of the device.
    In single GPU systems, use the value `0`.

    Example Value: `0`
    """

    interface: Optional[str]
    """
    The tool used to retrieve the informatoin listed in this entry.

    Example Value: `nvml`
    """

    clock_rate: Optional[float]  #:
    max_clock_rate: Optional[float]  #:
    memory_clock_rate: Optional[float]  #:
    memory_max_clock_rate: Optional[float]  #:

    extra: Extra = {}
    """
    Additional data the client believes is relevant.
    """


class System(Base):
    """
    Information about the system.

    Can be created automatically from the current system using NRSU.
    """

    _aliases = {
        "architecture": "cpu_arch",
        "mem_total__MB": "memory_total",
        "mem_total__MiB": "memory_total",
        "mem_total": "memory_total",
        "nvidia_driver_version": "gpu_driver_version",
        "product_name": "device_product_name",
        "ram_size": "memory_total",
        "shorthand_system_name": "short_name",
        "system_product_name": "device_product_name",
        "system_short_name": "short_name",
    }
    _duplicate_as = {
        Text: [
            "baseboard_name",
            "device_product_name",
            "os_name",
            "os_version",
            "short_name",
            "user",
        ]
    }

    @staticmethod
    def from_current(include_env=None):
        """
        Return an instance with data from the current machine using NRSU.

        :param include_env: List of environment variables to whitelist, or `True` to include all
        """
        from .utils.nrsu import get_system

        return get_system(include_env=include_env)

    # Public API
    baseboard_name: Optional[str]
    """
    Name of the motherboard.

    Example Value: `"X11SPG-TF"`
    """

    device_product_name: Optional[str]  #:
    env: FlatDict = {}  #: Environment variables
    gpu_driver_version: Optional[str]  #:
    hostname: Optional[str]  #:
    ip: Optional[str]  #:
    memory_configured_speed: Optional[str]  #:
    memory_total: Optional[int]  #:
    os_name: Optional[str]  #:
    os_version: Optional[str]  #:
    short_name: Optional[str]  #:
    total_memory_dimm_count: Optional[int]  #:
    user: Optional[str]  #:


class Workload(Base):
    """
    Information about what we are actually profiling. By default includes only
    fields for the command being executed, arguments and version. Can be
    extended to include additional fields by creating a subclass.
    """

    _type = None
    _duplicate_as = {Text: ["cmd", "args", "path", "vcs_repo", "vcs_branch"]}

    # @root_validator
    # def check_type(cls, values):
    #     # For subclasses, ensure there's a `_type`
    #     if values.get("_type"):
    #         raise ValueError("Can't specify a _type for instances")
    #     return values

    # Public API
    id: Optional[str]  #:

    name: Optional[str]  #: Name for this workload
    cmd: Optional[str]  #:
    """
    Command a user can run to reproduce the workload locally. Ideally, this
    should be a docker command.

    Example Value: `"docker run sc-hw-artf.nvidia.com/compute-docker/tensorrt-base:master-native-x86_64-ubuntu18.04-cuda11.0" /some/command`
    """

    args: Optional[str]  #:
    path: Optional[str]  #:
    """
    Workload runtime relative path.

    Example Value: `"/home/npengra/example/"`
    """

    version: Optional[str]  #:
    """
    Version of tool being run.

    For instance, for the TensorRT team has multiple TRT versions they run tests
    on. The version of TRT is recorded here.

    Example Value: `"7.9.0.0"`
    """

    vcs_repo: Optional[str]  #: Git remote URL or P4 location
    vcs_commit: Optional[str]  #: Git commit hash or P4 changelist
    vcs_branch: Optional[str]  #:
    vcs_extra: Dict[str, Any] = {}  #:

    ci_url: Optional[str]
    """
    URL for the CI pipeline or job that generated this record.

    Example Value: `https://gitlab-master.nvidia.com/compute/infrastructure/cnexus-perfdb/-/pipelines/2447072`
    """

    lib_versions: FlatDict = {}  #:


class Asset(Base):
    """
    Asset, representing an external asset such as a log file or report.
    """

    # Public API
    path: str
    """
    The object path of the asset if stored remotely, or the local path to the 
    object if stored locally
    """

    filesize: Optional[int]
    name: Optional[str]

    url: Optional[str]
    """
    The http url that can be used to retrieve the asset from remote storage. 
    Automatically generated by the API
    """

    upload_status: Optional[bool]
    """
    Indicates if this asset has been successfully uploaded or not. Will only
    used if remote=True
    """

    remote: Optional[bool]
    """
    Indicates if this asset should be stored remotely or locally. Defaults to
    True
    """

    @validator("name", pre=True, always=True)
    def set_default_name(cls, v, values):
        "The filename, extracted from the path"
        return v or values["path"].split("/")[-1]

    @validator("filesize", pre=True, always=True)
    def set_default_filesize(cls, v, values):
        file = Path(values["path"])
        if not file.exists():
            return None
        return file.stat().st_size

    @validator("remote", pre=True, always=True)
    def set_default_remote(cls, v, values):
        if v is None:
            return True
        return v


class Record(Base):
    """
    Base class shared by all Record/Result classes. Represents the
    result of the execution of a Workload, including information about the
    System, CPU and GPU it was executed on.

    Contains 3 default Measurement objects: duration, latency and throughput,
    each with a total, mean, min/max, etc. Also includes 3 additional float
    attributes that describe the total joules used to process the workload for
    the CPU, GPU and total system.

    This class defines the minimum information needed to store data in PerfDB.

    `group` can be used to group multiple results
    (eg. multiple layer results belonging to the same network run)
    """

    _type = None
    _duplicate_as = {Text: ["tags"]}

    @validator("id", pre=True, always=True, check_fields=False)
    def set_default_id(cls, v):
        return _random_uuid(cls, v)

    @property
    def s_type(self):
        return self._type

    @root_validator
    def check_type(cls, values):
        # For subclasses, ensure there's a `_type`
        if not cls._type:
            raise ValueError("No _type specified for this Result class.")
        return values

    @validator("client")
    def validate_client(cls, v):
        return _client_validator(v)

    @property
    def ts_created(self):
        return self.timestamp

    id: Optional[str]  #: Will be assigned a random UUID if missing
    client: str  #:

    timestamp: datetime.datetime
    """
    Timestamp noting the time when this record was created.
    For example, the time when a test was run to produce these results.
    """

    group: Optional[str]  #: Used to group multiple records
    tags: List[str] = []  #:
    comment: Optional[str]  #:

    # Metadata
    invalid: Optional[bool]  #:


class MetricRecord(Record):
    """
    Metrics which can be captured in parallel to a :py:class:`~PerformanceResult`.

    Eg. GPU usage over time when running a given workload.
    """

    @validator("result_id", pre=True, always=True)
    def validate_result_id(cls, v):
        if isinstance(v, PerformanceResult):
            return v.id
        elif isinstance(v, str):
            return v
        elif v:
            raise ValueError("result_id is not str or PerformanceResult")

    @validator("component_id", pre=True, always=True)
    def validate_component_id(cls, v):
        if v is None:
            return None
        if isinstance(v, Gpu):
            return v.uuid
        elif isinstance(v, (str, int)):
            return str(v)
        # else:
        #     raise ValueError("component_id is not str or GPU")

    _type = "metric"

    result_id: Optional[str]
    """
    Reference to an existing :py:attr:`~PerformanceResult` record's
    :py:attr:`~PerformanceResult.id`.

    Will be automatically assigned when this record is added to a :py:attr:`~PerformanceResult.metrics.
    Otherwise it can assigned either a direct reference to a :py:attr:`~PerformanceResult` or to the
    :py:attr:`~PerformanceResult.id` field.
    """

    component_id: Optional[str]
    """
    A reference to a specific piece of hardware if there could be multiple. E.g.
    this could reference a GPU ID in a multi GPU system, or a CPU index in a
    multi CPU system. Can be passed a GPU reference directly.
    """

    seq_id: Optional[int]
    """
    A sequence number for this metric record. For non-timeseries records this is filled automatically by the API, but can be overriden.
    """

    timestamp: Optional[datetime.datetime]
    """
    The timestamp for this metric, in case this is a timeseries metric. Can be omited otherwise.
    """

    name: MetricType
    """
    The component which this metric is covering.
    """

    value: float
    """
    The value of this measurement. Unit depends on the MetricType.
    """

    @classmethod
    def bulk_create(
        cls,
        result_id: Union[str, Record],
        client: str,
        timestamp: datetime.datetime,
        data: Dict[Union[MetricType, str], float],
        comment: Optional[str] = None,
        tags: List[str] = [],
        group: Optional[str] = None,
        component_id: Optional[str] = None,
    ) -> List["MetricRecord"]:
        """
        A simple method the converts a dictionary containing names to values to
        a list of `MetricRecord`s.

        For instance, the variable `metric_records` below will contain a list of
        `MetricRecord`s that represent the same data in the `data` dictionary:

        ```py
        data = {
            MetricType.gpu_clock: 123,
            'cpu_clock': 34,
            # ... others
        }
        metric_records = MetricRecord.bulk_create(
            result_id=result,
            client="myself",
            timestamp=now(),
            data=data,
        )
        ```


        """
        meta = {
            "result_id": result_id,
            "client": client,
            "timestamp": timestamp,
            "comment": comment,
            "tags": tags,
            "group": group,
            "component_id": component_id,
        }

        return [
            cls(name=name, value=value, **meta)
            for name, value in data.items()
            if value is not None
        ]


class PerformanceResult(Record):
    """
    A single performance measurement result, including System/GPU and Workload information.
    """

    _duplicate_as = {Text: ["run_name"]}

    @property
    def _type(self):
        """Type will come either from the Workload or from the Result class"""
        return self.workload._type or "result"

    @root_validator
    def check_type(cls, values):
        # For subclasses, ensure there's a `_type`
        if values.get("_type"):
            raise ValueError("Can't specify a _type for instances")
        return values

    @validator("group", pre=True, always=True)
    def set_default_group(cls, v):
        return _random_uuid(cls, v)

    @validator("duration", pre=True)
    def validate_duration(cls, v):
        return _measurement_validator(cls, v)

    @validator("latency", pre=True)
    def validate_latency(cls, v):
        return _measurement_validator(cls, v)

    @validator("throughput", pre=True)
    def validate_throughput(cls, v):
        return _measurement_validator(cls, v)

    def dict(self, nvdataflow=False, **kwargs):
        res = super().dict(nvdataflow, **kwargs)

        # add gpu.l_device_count
        if res.get("gpu"):
            for g in res["gpu"]:
                g["l_device_count"] = len(res["gpu"])

        return res

    @property
    def s_gpu(self):
        """GPU string: `<device>` if there's only one, `<# gpus>x <device>` otherwise."""
        if len(self.gpu) and all(
            g.device_product_name == self.gpu[0].device_product_name for g in self.gpu
        ):
            return (
                self.gpu[0].device_product_name
                if len(self.gpu) == 1
                else f"{len(self.gpu)}x {self.gpu[0].device_product_name}"
            )

    @property
    def gpu_power_efficiency(self) -> Optional[Measurement]:
        "Efficiency of GPU power usage, calculated via throughput / total GPU energy consumed"
        return _component_efficiency(self, "gpu")

    @property
    def cpu_power_efficiency(self) -> Optional[Measurement]:
        "Efficiency of CPU power usage, calculated via throughput / total CPU energy consumed"
        return _component_efficiency(self, "cpu")

    @property
    def system_power_efficiency(self) -> Optional[Measurement]:
        "Efficiency of system power usage, calculated via throughput / total system energy consumed"
        return _component_efficiency(self, "system")

    has_metrics: bool = False
    """
    CNEXUS Dashboard flag: used for filtering out results w/ no metrics.
    Set to true if `MetricRecord` instances where the `result_id` is set to
    this `id` are being uploaded to perfdb.
    """

    id: Optional[str]  #: Will be assigned a random UUID if missing

    run_name: Optional[str]  #:

    workload: Workload
    """
    Workload that generated the measurement.

    Ideally, this is a 1:1 to relationship, 1 workload per result. Not be used
    as a foreign key.
    """

    system: Optional[System]
    """
    System the Workload ran on; can be created.

    Can be imported from the current system with :py:func:`System.from_current`.
    """

    gpu: List[Gpu] = []
    """
    GPU(s) the Workload used.

    Can be imported from the current system with :py:func:`Gpu.from_current`
    """

    cpu: Optional[Cpu]
    """
    CPU the Workload used.

    Can be imported from the current system with :py:func:`Cpu.from_current`
    """

    """
    Measurement values
    Also accepts `Float` values, which will be converted to `Measurement(total=value)`

    Measurement objects must have mean values for efficiencies to be calculated.
    """
    duration: Optional[Measurement]  #:
    latency: Optional[Measurement]  #:
    throughput: Optional[Measurement]  #:

    """
    Component power values
    Measures the average power consumption of the gpu, cpu and total system in
    watts.

    Measurement objects must have mean values for efficiencies to be calculated.
    """
    gpu_power: Optional[Measurement]  #: GPU energy consumption in watts
    cpu_power: Optional[Measurement]  #: CPU energy consumption in watts
    system_power: Optional[Measurement]  #: System energy consumption in watts

    """
    Misc assets associated with a performance result.
    Ex. Log files, dnnx files,
    """
    assets: List[Asset] = []  #:

    metrics: List[MetricRecord] = []
    """
    List of metrics associated with this performance result.
    """
