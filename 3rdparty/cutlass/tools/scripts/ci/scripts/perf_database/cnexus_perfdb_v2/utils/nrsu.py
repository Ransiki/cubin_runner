import re
import logging

from ..models import Gpu, Cpu, System

logger = logging.getLogger(__name__)


def get_system(include_env=None):
    params = {}
    for name in ("cpu", "os", "gpu", "system"):
        params[f"{name}_properties"] = _get_properties(name)

    # Get hostname; TODO: move to nrsu
    try:
        import socket

        params["hostname"] = socket.gethostname()
    except Exception:
        pass

    return parse_system(**params, include_env=include_env)


def get_gpus():
    gpu_properties = _get_properties("gpu") or []
    device_count = len(gpu_properties) if gpu_properties is not None else None
    gpus = [
        parse_gpu(gpu_data, device_count=device_count) for gpu_data in gpu_properties
    ]
    return [gpu for gpu in gpus if gpu]


def get_cpu():
    cpu_properties = _get_properties("cpu")
    return parse_cpu(cpu_properties)


def parse_cpu(cpu_properties, system_properties=None):
    """Create a :py:class:`Cpu <cnexus_perfdb.models.Cpu>` object from an NRSU CPU properties object."""
    cpu_data = _from_object(cpu_properties)
    system_data = _from_object(system_properties)

    cpu_data = dict(
        # TODO: short_name
        **{
            k: cpu_data.get(k) or system_data.get(k)
            for k in [
                "product_name",
                "architecture",
                "cores_per_socket",
                "cpu_count",
                "shorthand_cpu_name",
            ]
        },
    )
    return Cpu(**cpu_data)


def parse_gpu(gpu_properties, device_count=None):
    """
    Create a :py:class:`Gpu <cnexus_perfdb.models.Gpu>` object from s list of NRSU GPU properties objects.
    """
    nrsu_data = _from_object(gpu_properties)
    nrsu_cuda_attrs = nrsu_data.get("cuda_attributes") or {}
    gpu_data = dict(
        device_count=device_count,
        memory_total=nrsu_data.get("mem_total__MiB") or nrsu_data.get("mem_total__MB"),
        vbios=nrsu_data.get("vbios") or nrsu_data.get("vbios_version"),
        # GPU/Memory clocks from cuda attributes are in KHz, so convert to MHz
        **{
            k: nrsu_cuda_attrs[k] / 1000
            for k in ["clock_rate", "memory_clock_rate"]
            if nrsu_cuda_attrs.get(k)
        },
        **{
            k: nrsu_data.get(k)
            for k in [
                "architecture",
                "chip",
                "compute_capability",
                "device_product_name",
                "gpu_max_clock__MHz",
                "index",
                "interface",
                "mem_max_clock__MHz",
                "multiprocessor_count",
                "pci_bus_id",
                "pci_device_id",
                "power_limit__W",
                "serial_number",
                "shorthand_system_name",
                "uuid",
            ]
        },
    )
    return Gpu(**gpu_data)


def parse_system(
    cpu_properties,
    os_properties,
    gpu_properties,
    system_properties=None,
    ip=None,
    hostname=None,
    include_env=None,
    **kwargs,
):
    """
    Create a :py:class:`System <cnexus_perfdb.models.System>` object from NRSU CPU, OS and GPU properties objects.

    Does *not* include :py:class:`GPUs <cnexus_perfdb.models.Gpu>`; they need to be created with :py:func:`parse_gpu() <cnexus_perfdb.utils.nrsu.parse_gpu>` and assigned to the :py:class:`System <cnexus_perfdb.models.System>`.gpus list.
    """
    # whitelist env vars?
    if os_properties and os_properties.get("env") and include_env is not True:
        if not include_env:
            os_properties["env"] = None
        elif isinstance(include_env, list):
            all_env_vars = os_properties["env"]

            # Create a regex to match any of the env vars in `include_env`;
            # to support `*`, replace it with `.+`;
            # Eg. ['a', 'b', 'c*'] => `^(a|b|c.+)$`
            env_regex = re.compile(
                rf"^({'|'.join(item.replace('*', '.+') for item in include_env)})$"
            )

            os_properties["env"] = {
                k: v for k, v in all_env_vars.items() if env_regex.match(k)
            }

    cpu_data = _from_object(cpu_properties)
    os_data = _from_object(os_properties)
    memory_data = _from_object(os_properties.get("memory"))
    system_data = _from_object(system_properties)

    nvidia_driver_version = (
        gpu_properties[0].get("nvidia_driver_version")
        if gpu_properties and len(gpu_properties)
        else None
    )

    data = dict(
        hostname=hostname,
        ip=ip or os_data.get("ip"),
        gpu_driver_version=nvidia_driver_version,
        memory_total=float(memory_data["mem_total__kB"]) / 1024
        if memory_data.get("mem_total__kB")
        else 0,
        os_name=os_data.get("name"),
        os_version=os_data.get("version"),
    )

    for (source, fields) in [
        (os_data, ["user", "env"]),
        (cpu_data, ["baseboard_name", "system_product_name"]),
        (
            system_data,
            [
                "memory_configured_voltage",
                "memory_configured_speed",
                "memory_type",
                "memory_size",
                "total_memory_dimm_count",
                "data_memory_width",
                "total_memory_width",
                "shorthand_system_name",
                "system_product_name",
                "baseboard_name",
            ],
        ),
    ]:
        data.update({k: source[k] for k in fields if source.get(k)})

    return System(**data, **kwargs)


def parse_gpu_metric(gpu_metric_properties, **additional_data):
    from ..models import MetricRecord, MetricType

    gpu_metrics = []

    nrsu_data = {**_from_object(gpu_metric_properties), **additional_data}
    mapping = {
        MetricType.gpu_clock: "gpu_clock__MHz",
        MetricType.gpu_util: "gpu_utilization__pct",
        MetricType.gpu_temp: "gpu_temperature__C",
        MetricType.gpu_fan_speed: "fan_speed__pct",
        MetricType.gpu_graphics_clock: "graphics_clock__MHz",
        MetricType.gpu_power_draw: "power_draw__W",
        MetricType.gpu_memory_clock: "memory_clock__MHz",
        MetricType.gpu_memory_util: "memory_utilization__pct",
        MetricType.gpu_memory_temp: "memory_temperature__C",
        MetricType.gpu_decoder_util: "decoder_utilization__pct",
        MetricType.gpu_encoder_util: "encoder_utilization__pct",
    }
    gpu_data = {k: nrsu_data.get(v) for k, v in mapping.items()}
    meta_data = {
        v: nrsu_data.get(k)
        for k, v in {
            "timestamp": "timestamp",
            "client": "client",
            "gpu_uuid": "component_id",
            "uuid": "component_id",
            "result_id": "result_id",
            "result": "result_id",
        }.items()
        if nrsu_data.get(k)
    }

    for k, v in gpu_data.items():
        if v is not None:
            gpu_metrics.append(MetricRecord(name=k, value=v, **meta_data))

    return gpu_metrics


def _get_properties(name):
    import nrsu

    try:
        return getattr(nrsu, name, None).get_properties()
    except Exception:
        logger.warning("Unable to retrieve %s properties from NRSU, skipping", name)
    return None


def _from_object(nrsu_obj):
    if nrsu_obj and not isinstance(nrsu_obj, dict):
        raise ValueError("Not an NRSU object.")
    if nrsu_obj and "dictitems" in nrsu_obj:
        nrsu_obj = nrsu_obj["dictitems"]
    if nrsu_obj is None:
        nrsu_obj = {}
    return nrsu_obj
