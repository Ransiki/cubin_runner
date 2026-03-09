import uuid
from functools import reduce
import operator
from typing import Optional

from ..base import Base
from ..models import PerformanceResult, Cpu, Gpu, System
from ..ext.deep_learning.models import (
    DLNetworkWorkload,
    DLOperatorWorkload,
)


class CustomWorkload(Base):
    """custom class that adds all clients' extra fields"""

    # flow.dnn
    flow_id: Optional[str]
    implementation_selection_method: Optional[str]

    # mlperf_inference_silicon
    accuracy_level: Optional[str]
    optimization_level: Optional[str]
    inference_server: Optional[str]
    scenario: Optional[str]  # ie. mode
    input_format: Optional[str]
    use_graphs: Optional[bool]


class CustomNetworkWorkload(CustomWorkload, DLNetworkWorkload):
    pass


class CustomOperatorWorkload(CustomWorkload, DLOperatorWorkload):
    pass


class CustomResult(PerformanceResult):
    # mlperf_inference_silicon
    valid: Optional[bool]


MAPPING = {
    PerformanceResult: {
        "id": "s_id",
        "client": "s_client",
        "timestamp": "ts_created",
        "group": "s_group",
        "tags": "session.workload.result.s_tags",
        "run_name": "s_name",
        "duration": "session.workload.result.d_value",
        # mlperf-inf
        "comment": "session.workload.result.extra.s_comment",
    },
    Cpu: {
        "device_product_name": "session.system.s_cpu_product_name",
        "short_name": "session.system.s_cpu_short_name",
        "architecture": "session.system.s_cpu_arch",
        "device_count": "session.system.l_cpu_count",
    },
    System: {
        "baseboard_name": "session.system.s_baseboard_name",
        "env": "session.workload.env",
        "hostname": "session.system.s_hostname",
        "ip": "session.system.s_ip",
        "memory_total": "session.system.l_mem_total",
        "gpu_driver_version": "session.system.s_nvidia_driver_version",
        "os_name": "session.system.s_os_name",
        "os_version": "session.system.s_os_version",
        "system_product_name": "session.system.s_system_product_name",
        "system_short_name": "session.system.s_system_short_name",
        "user": "session.s_user",
        "data_memory_width": "session.system.s_data_memory_width",
        "memory_size": "session.system.s_memory_size",
        "total_memory_dimm_count": "session.system.s_total_memory_dimm_count",
        "memory_type": "session.system.s_memory_type",
        "memory_configured_speed": "session.system.s_memory_configured_speed",
        "memory_configured_voltage": "session.system.s_memory_configured_voltage",
    },
    DLNetworkWorkload: {
        "id": "session.workload.s_id",
        "name": "session.workload.s_name",
        "cmd": "session.workload.s_cmd",
        "args": "session.workload.s_args",
        "path": "session.workload.s_path",
        "version": "session.workload.s_version",
        "vcs_repo": "session.workload.s_vcs_repo",
        "vcs_commit": "session.workload.s_vcs_commit",
        "vcs_branch": "session.workload.s_vcs_branch",
        "vcs_extra": "session.workload.vcs_extra",
        "lib_versions": "session.workload.lib_versions",
        "network": "session.workload.result.s_network",
        "network_variant": "session.workload.result.s_network_variant",
        "network_description_format": "session.workload.result.s_network_description_format",
        "mode": "session.workload.result.s_mode",
        "platform": "session.workload.result.s_platform",
        "gpus": "session.s_gpu",
        # backwards-compatible precision fields
        "storage_precision": "session.workload.result.implementation.s_storage_precision",
        "compute_precision": "session.workload.result.implementation.s_compute_precision",
        # flow.dnn
        "flow_id": "session.workload.extra.s_flow_id",
        "implementation_selection_method": "session.workload.result.s_optimization_strategy",
        # mlperf-inf
        "accuracy_level": "session.workload.s_accuracy_level",
        "optimization_level": "session.workload.s_optimization_level",
        "inference_server": "session.workload.s_inference_server",
        "scenario": "session.workload.result.extra.s_mode",
        "input_format": "session.workload.result.extra.s_input_format",
        "use_graphs": "session.workload.result.extra.s_use_graphs",
        # DLOperatorWorkloadBase
        "bias_layout": "session.workload.result.implementation.s_c_layout",
        "bias_numeric_type": "session.workload.result.implementation.s_c_precision",
        "implementation_id": "session.workload.result.implementation.s_id",
        "implementation_name": "session.workload.result.implementation.s_name",
        "input_channels": "session.workload.result.operator.inputs.C",
        "input_dim_d": "session.workload.result.operator.inputs.D",
        "input_dim_h": "session.workload.result.operator.inputs.H",
        "input_dim_w": "session.workload.result.operator.inputs.W",
        "input_layout": "session.workload.result.implementation.s_a_layout",
        "input_numeric_type": "session.workload.result.implementation.s_a_precision",
        "operator_name": "session.workload.result.operator.s_name",
        "operator_type": "session.workload.result.operator.s_type",
        "output_channels": "session.workload.result.operator.outputs.C",
        "output_dim_d": "session.workload.result.operator.outputs.D",
        "output_dim_h": "session.workload.result.operator.outputs.H",
        "output_dim_w": "session.workload.result.operator.outputs.W",
        "output_layout": "session.workload.result.implementation.s_d_layout",
        "output_numeric_type": "session.workload.result.implementation.s_d_precision",
        "phase": "session.workload.result.implementation.s_phase",
        "primitive": None,
        "weights_layout": "session.workload.result.implementation.s_b_layout",
        "weights_numeric_type": "session.workload.result.implementation.s_b_precision",
        # GEMMLikeOperatorWorkload
        "a_tensor_layout": "session.workload.result.implementation.s_a_layout",
        "a_tensor_numeric_type": "session.workload.result.implementation.s_a_precision",
        "accumulator_numeric_type": "session.workload.result.implementation.s_accumulator_precision",
        "alpha": "session.workload.result.implementation.d_alpha",
        "b_tensor_layout": "session.workload.result.implementation.s_b_layout",
        "b_tensor_numeric_type": "session.workload.result.implementation.s_b_precision",
        "batch_size": "session.workload.result.implementation.l_batch_size",
        "beta": "session.workload.result.implementation.d_beta",
        "c_tensor_layout": "session.workload.result.implementation.s_c_layout",
        "c_tensor_numeric_type": "session.workload.result.implementation.s_c_precision",
        "cask_implementation": "session.workload.result.implementation.b_cask_supported",
        "cta_occupancy_per_sm": "session.workload.result.implementation.l_cta_occupancy_per_sm",
        "cta_raster_order": "session.workload.result.implementation.s_cta_raster_order",
        "cta_tile_size": "session.workload.result.implementation.s_cta_tile_size",
        "d_tensor_layout": "session.workload.result.implementation.s_d_layout",
        "d_tensor_numeric_type": "session.workload.result.implementation.s_d_precision",
        "gemm_k": "session.workload.result.operator.inputs.GEMM_K",
        "gemm_m": "session.workload.result.operator.inputs.GEMM_M",
        "gemm_n": "session.workload.result.operator.inputs.GEMM_N",
        "input_data_range": "session.workload.result.implementation.d_input_data_range",
        "inter_cta_k_split_factor": "session.workload.result.implementation.l_inter_cta_k_split_factor",
        "intra_cta_k_split_factor": "session.workload.result.implementation.l_intra_cta_k_split_factor",
        "k_block_size": "session.workload.result.implementation.l_k_block_size",
        "kernel_modular": "session.workload.result.implementation.l_kernel_modular",
        "kernel_name": "session.workload.result.implementation.s_kernel_name",
        "kernel_signature": "session.workload.result.implementation.s_kernel_signature",
        "math_input_a_numeric_type": "session.workload.result.implementation.s_math_input_precision",
        "math_input_b_numeric_type": "session.workload.result.implementation.s_math_input_precision",
        "math_instruction": "session.workload.result.implementation.s_math_instruction",
        "min_smem_carveout": "session.workload.result.implementation.l_min_smem_carveout",
        "sparse_math_enabled": None,
        "warp_tile_size": "session.workload.result.implementation.s_warp_tile_size",
        # ConvolutionWorkload
        "convolution_algorithm": None,
        "dilation_d": "session.workload.result.operator.parameters.dilation_d",
        "dilation_h": "session.workload.result.operator.parameters.dilation_h",
        "dilation_w": "session.workload.result.operator.parameters.dilation_w",
        "filter_d": "session.workload.result.operator.weights.f_d",
        "filter_h": "session.workload.result.operator.weights.f_h",
        "filter_w": "session.workload.result.operator.weights.f_w",
        "num_groups": "session.workload.result.operator.parameters.num_groups",
        "padding_d": "session.workload.result.operator.parameters.pad_d",
        "padding_h": "session.workload.result.operator.parameters.pad_h",
        "padding_w": "session.workload.result.operator.parameters.pad_w",
        "stride_d": "session.workload.result.operator.parameters.stride_d",
        "stride_h": "session.workload.result.operator.parameters.stride_h",
        "stride_w": "session.workload.result.operator.parameters.stride_w",
    },
}

MEASUREMENT_MAPPING = {
    "min": "d_min",
    "mean": "d_mean",
    "max": "d_max",
    "target": "d_target",
    "percentiles": "percentiles",
    "value": "d_value",
}

# For GPUs we need to handle each structure individually, so don't use prefixes
GPU_MAPPING = {
    "device_product_name": "s_device_product_name",
    "chip": "s_chip",
    "compute_capability": "d_compute_capability",
    "uuid": "s_uuid",
    "pci_bus_id": "s_pci_bus_id",
    "pci_device_id": "s_pci_device_id",
    "power_limit": "d_power_limit",
    "serial_number": "s_serial_number",
    "memory_total": ["d_mem_total", "d_memory_total"],
    "multiprocessor_count": "l_multiprocessor_count",
    "vbios": "s_vbios",
    "index": "l_index",
    "interface": "s_interface",
    "clock_rate": "d_clock_rate",
    "max_clock_rate": "d_max_clock_rate",
    "memory_clock_rate": "d_memory_clock_rate",
    "memory_max_clock_rate": "d_memory_max_clock_rate",
}


def get_by_path(root, items, default=None):
    """Access a nested object in root by item sequence."""
    try:
        return reduce(operator.getitem, items, root)
    except KeyError:
        pass
    return default


def get_nested(doc, key, format=None):
    """get value from a nested dict that matches the first `fields` key"""
    res = next(
        (
            get_by_path(doc, k.split("."))
            for k in (key if isinstance(key, list) else [key])
        ),
        None,
    )

    if format:
        res = format(res)
    return res


def parse_v1_network_or_operator(doc_json):

    if not doc_json:
        return None

    doc = doc_json

    is_operator = get_by_path(doc_json, "session.workload.result.operator".split("."))
    data = {
        cls_: {
            new_field: get_nested(doc, old_fields)
            for new_field, old_fields in mapping.items()
            if old_fields and get_nested(doc, old_fields)
        }
        for cls_, mapping in MAPPING.items()
        if mapping
    }

    # training_framework?
    if data[DLNetworkWorkload].get("mode") == "training":
        data[DLNetworkWorkload]["training_framework"] = get_nested(
            doc, "session.workload.s_framework"
        )

    # generate a network result ID
    # fields_used_for_network_id = [
    #     "group",
    #     "network",
    #     "network_variant",
    #     "batch_size",
    #     "storage_precision",
    #     "compute_precision",
    #     "mode",
    #     "framework",
    #     "gpus",
    #     "platform",
    # ]
    # field_data_for_network_id = reduce(
    #     lambda acc, f: f"{acc}-{data[PerformanceResult].get(f) or data[DLNetworkWorkload].get(f) or ''}",
    #     fields_used_for_network_id,
    #     "",
    # )

    # network_generated_id = uuid.uuid5(uuid.NAMESPACE_DNS, field_data_for_network_id)

    # if not is_operator:
    #     data["s_id"] = network_generated_id
    # else:
    #     data["network_result_id"] = network_generated_id

    gpus_data = [
        {
            new_field: get_nested(gpu, old_fields)
            for new_field, old_fields in GPU_MAPPING.items()
            if old_fields
        }
        for gpu in get_by_path(doc, "session.gpus".split("."), [])
    ]
    measurements_data = {
        m: {
            new_field: get_nested(doc, f"session.workload.result.{m}.{old_fields}")
            for new_field, old_fields in MEASUREMENT_MAPPING.items()
            if old_fields
            and get_nested(doc, f"session.workload.result.{m}.{old_fields}")
        }
        for m in ["latency", "throughput"]
    }

    # MLPerf-Inf
    measurements_data["valid"] = get_nested(
        doc,
        "session.workload.result.extra.s_result_validity",
        format=lambda x: None if not x else x == "VALID",
    )

    # custom latency
    if not get_nested(measurements_data, "latency.value"):
        measurements_data["latency"]["value"] = get_nested(
            doc,
            "session.workload.result.latency.percentiles.90"
            if get_nested(doc, "session.workload.result.extra.s_mode")
            == "single-stream"
            else "session.workload.result.latency.percentiles.99",
        )

    measurements_data["latency"]["extra"] = {
        "d_achieved": get_nested(
            doc,
            "session.workload.result.extra.d_latency_achieved_ns",
            format=lambda x: x / 1000000000 if x else x,
        ),
        "d_single_stream_expected": get_nested(
            doc,
            "session.workload.result.extra.d_gpu_single_stream_expected_latency_ns",
            format=lambda x: x / 1000000000 if x else x,
        ),
    }

    # custom throughput
    measurements_data["throughput"]["extra"] = {
        "d_with_loadgen_overhead": get_nested(
            doc,
            "session.workload.result.extra.d_qps_with_loadgen_overhead",
        ),
        "d_without_loadgen_overhead": get_nested(
            doc,
            "session.workload.result.extra.d_qps_without_loadgen_overhead",
        ),
        "d_pairs": get_nested(
            doc,
            "session.workload.result.extra.d_pairs_per_sec",
        ),
        "d_offline_expected": get_nested(
            doc,
            "session.workload.result.extra.d_gpu_offline_expected_qps",
        ),
        "d_server_target": get_nested(
            doc,
            "session.workload.result.extra.d_server_target_qps",
        ),
    }

    return CustomResult(
        **data[PerformanceResult],
        **measurements_data,
        cpu=Cpu(**data[Cpu]),
        system=System(**data[System]),
        gpu=[Gpu(**gpu_data) for gpu_data in gpus_data],
        workload=CustomNetworkWorkload(**data[DLNetworkWorkload])
        if not is_operator
        else CustomOperatorWorkload(**data[DLNetworkWorkload]),
    )
