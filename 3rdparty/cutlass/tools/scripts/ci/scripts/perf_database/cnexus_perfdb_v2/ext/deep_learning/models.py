from typing import List, Optional

from pydantic import validator, root_validator

from ...models import Workload
from ...base import Text
from .enums import (
    Instruction,
    Network,
    NetworkDescriptionFormat,
    NetworkVariant,
    NetworkApplication,
    OperatorType,
    Mode,
    Framework,
    Platform,
    Precision,
    Phase,
    TensorLayout,
    GemmAlgorithm,
    ConvolutionAlgorithm,
)
from .validators import (
    batch_norm_validator,
    convolution_validator,
    dl_workload_validator,
    elementwise_validator,
    fully_connected_validator,
    gemm_like_validator,
    inference_network_validator,
    network_base_validator,
    operator_base_validator,
    training_network_validator,
    cask_rollup_network_validator,
)


class DLWorkload(Workload):
    _root_validator = root_validator()(dl_workload_validator)
    _type = "network"
    _duplicate_as = {
        Text: ["mode", "network_variant", "network", "platform", "training_framework"]
    }

    network: Network  #:
    batch_size: float  #: mean if multiple

    # Optional
    batch_size_distribution: List[
        int
    ] = (
        []
    )  #: a 1D array representing a histogram of batch sizes for cases with variable batch size
    cask_branch: Optional[str]  #:
    cask_commit: Optional[str]  #:
    cask_version: Optional[str]  #:
    cudnn_branch: Optional[str]  #:
    cudnn_commit: Optional[str]  #:
    cudnn_version: Optional[str]  #:
    cuda_version: Optional[str]  #: Track CUDA version
    mode: Optional[Mode]  #:
    network_variant: Optional[NetworkVariant]  #:
    network_application: Optional[NetworkApplication]  #:
    dlsim_methodology: Optional[str]  #: DLSim methodology used (eg. `proj`)
    platform: Optional[Platform]  #:
    training_framework: Optional[Framework]  #:
    framework: Optional[Framework]  #:


# Network Workloads


class DLNetworkWorkloadBase(DLWorkload):
    _root_validator = root_validator()(network_base_validator)

    def __init__(self, *args, **kwargs):
        ps = [
            kwargs.pop("storage_precision", None),
            kwargs.pop("compute_precision", None),
        ]
        super(DLNetworkWorkloadBase, self).__init__(*args, **kwargs)
        for p in ps:
            self.set_precision(p)

    def __setattr__(self, name, value):
        if name in ("compute_precision", "storage_precision"):
            self.set_precision(value)
        else:
            return super(DLNetworkWorkloadBase, self).__setattr__(name, value)

    def set_precision(self, value):
        if value and hasattr(self, f"{value}_enabled"):
            setattr(self, f"{value}_enabled", True)

    @property
    def precisions(self) -> str:
        all_ps = sorted(
            p
            for p in ("bf16", "fp16", "fp32", "int4", "int8", "tf32")
            if getattr(self, f"{p}_enabled", False) is True
        )
        return "/".join(sorted(all_ps)) if all_ps else None

    tensor_layout: Optional[TensorLayout]  #:
    export_sparsity_enabled: Optional[bool]  #:
    bf16_enabled: Optional[bool]  #:
    fp16_enabled: Optional[bool]  #:
    fp32_enabled: Optional[bool]  #:
    int4_enabled: Optional[bool]  #:
    int8_enabled: Optional[bool]  #:
    tf32_enabled: Optional[bool]  #:
    fusions: List[str] = []  # Fusions enabled while running the workload


class InferenceNetworkWorkload(DLNetworkWorkloadBase):
    _root_validator = root_validator()(inference_network_validator)
    _duplicate_as = {Text: ["network_description_format", "inference_server"]}

    triton_version: Optional[str]  #:
    triton_commit: Optional[str]  #:
    triton_branch: Optional[str]  #:
    tensorrt_version: Optional[str]  #:
    tensorrt_commit: Optional[str]  #:
    tensorrt_branch: Optional[str]  #:
    inference_server: Optional[str]  # TODO: Enum?
    network_description_format: Optional[NetworkDescriptionFormat]  #:


class TrainingNetworkWorkload(DLNetworkWorkloadBase):
    _root_validator = root_validator()(training_network_validator)

    framework_version: Optional[str]  #:
    framework_commit: Optional[str]  #:
    framework_branch: Optional[str]  #:
    nccl_version: Optional[str]  #:
    nccl_commit: Optional[str]  #:
    nccl_branch: Optional[str]  #:


class CaskRollupNetworkWorkload(DLNetworkWorkloadBase):
    _root_validator = root_validator()(cask_rollup_network_validator)

    network_description_format: Optional[NetworkDescriptionFormat]  #:

    a_tensor_numeric_type: Optional[Precision]  #:
    accumulator_numeric_type: Optional[Precision]  #:
    b_tensor_numeric_type: Optional[Precision]  #:
    c_tensor_numeric_type: Optional[Precision]  #:
    d_tensor_numeric_type: Optional[Precision]  #:
    math_input_a_numeric_type: Optional[Precision]  #:
    math_input_b_numeric_type: Optional[Precision]  #:
    math_instruction: Optional[Instruction]  #:


# Operator Workloads


class DLOperatorWorkloadBase(DLNetworkWorkloadBase):
    _root_validator = root_validator()(operator_base_validator)
    _type = "operator"
    _duplicate_as = {Text: ["operator_name", "operator_type", "primitive", "phase"]}

    @validator("implementation_id", pre=True, always=True)
    def set_default_implementation_id(cls, v, values):
        """
        Set implementation ID to implementation name by default.

        Required since different flow.dnn implementations have the same name;
        others don't need to set an ID if they have unique names.
        """
        return v if v is not None else values.get("implementation_name")

    network_result_id: Optional[
        str
    ]  #: Result ID of the network result this record applies to
    operator_name: str  #: Network dependent name for the operator
    operator_type: Optional[OperatorType]  #: The DNNX equivalent operation type.
    primitive: Optional[str]  #: The network independent name for the operator
    phase: Optional[Phase]  #:
    implementation_id: Optional[str]  #:
    implementation_name: Optional[str]  #:
    input_operator_ids: List[
        str
    ] = []  #: list of workload IDs for operators that generate inputs for this one
    output_operator_ids: List[
        str
    ] = []  #: list of workload IDs for operators that generate outputs for this one
    input_channels: Optional[int]  #:
    input_dim_d: Optional[int]  #:
    input_dim_h: Optional[int]  #:
    input_dim_w: Optional[int]  #:
    input_layout: Optional[TensorLayout]  #:
    input_numeric_type: Optional[Precision]  #:
    weights_layout: Optional[TensorLayout]  #:
    weights_numeric_type: Optional[Precision]  #:
    bias_layout: Optional[TensorLayout]  #:
    bias_numeric_type: Optional[Precision]  #:
    output_channels: Optional[int]  #:
    output_dim_d: Optional[int]  #:
    output_dim_h: Optional[int]  #:
    output_dim_w: Optional[int]  #:
    output_layout: Optional[TensorLayout]  #:
    output_numeric_type: Optional[Precision]  #:


class GEMMLikeOperatorWorkload(DLOperatorWorkloadBase):
    _root_validator = root_validator()(gemm_like_validator)

    a_tensor_layout: Optional[TensorLayout]  #:
    a_tensor_numeric_type: Optional[Precision]  #:
    accumulator_numeric_type: Optional[Precision]  #:
    alpha: Optional[float]  #:
    b_tensor_layout: Optional[TensorLayout]  #:
    b_tensor_numeric_type: Optional[Precision]  #:
    beta: Optional[float]  #:
    c_tensor_layout: Optional[TensorLayout]  #:
    c_tensor_numeric_type: Optional[Precision]  #:
    cask_implementation: Optional[bool]  #:
    cta_occupancy_per_sm: Optional[int]  #:
    cta_raster_order: Optional[str]  #:
    cta_tile_size: Optional[str]  #:
    d_tensor_layout: Optional[TensorLayout]  #:
    d_tensor_numeric_type: Optional[Precision]  #:
    fused_activation_function: Optional[bool]  #:
    gemm_algorithm: Optional[GemmAlgorithm]  #:
    gemm_k: Optional[int]  #:
    gemm_m: Optional[int]  #:
    gemm_n: Optional[int]  #:
    input_data_range: Optional[float]  #:
    inter_cta_k_split_factor: Optional[int]  #:
    intra_cta_k_split_factor: Optional[int]  #:
    k_block_size: Optional[int]  #:
    k_block_stages: Optional[int]  #:
    kernel_modular: Optional[int]  #:
    kernel_name: Optional[str]  #: Generic human readable name of kernel
    kernel_signature: Optional[str]  #: Full C++ reference of the kernel
    math_input_a_numeric_type: Optional[Precision]  #:
    math_input_b_numeric_type: Optional[Precision]  #:
    math_instruction: Optional[Instruction]  #:
    min_smem_carveout: Optional[
        int
    ]  #: Minimum memory the kernel asked to allocate for this GEMM
    sparse_math_enabled: Optional[bool]  #:
    warp_tile_size: Optional[str]  #:


class ElementwiseOperatorWorkload(DLOperatorWorkloadBase):
    _root_validator = root_validator()(elementwise_validator)


class ConvolutionWorkload(GEMMLikeOperatorWorkload):
    _root_validator = root_validator()(convolution_validator)

    convolution_algorithm: Optional[ConvolutionAlgorithm]  #:
    dilation_d: Optional[int]  #:
    dilation_h: Optional[int]  #:
    dilation_w: Optional[int]  #:
    filter_d: Optional[int]  #:
    filter_h: Optional[int]  #:
    filter_w: Optional[int]  #:
    num_groups: Optional[int]  #:
    padding_d: Optional[int]  #:
    padding_h: Optional[int]  #:
    padding_w: Optional[int]  #:
    stride_d: Optional[int]  #:
    stride_h: Optional[int]  #:
    stride_w: Optional[int]  #:


class FullyConnectedWorkload(GEMMLikeOperatorWorkload):
    _root_validator = root_validator()(fully_connected_validator)


class BatchNormWorkload(GEMMLikeOperatorWorkload):
    _root_validator = root_validator()(batch_norm_validator)


""" User-Facing API """


class DLNetworkWorkload(
    InferenceNetworkWorkload, TrainingNetworkWorkload, CaskRollupNetworkWorkload
):
    pass


class DLOperatorWorkload(
    ConvolutionWorkload, FullyConnectedWorkload, BatchNormWorkload
):
    pass
