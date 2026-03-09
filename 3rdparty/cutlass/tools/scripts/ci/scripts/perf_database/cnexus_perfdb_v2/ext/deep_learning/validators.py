import warnings
import logging

from .enums import OperatorType

logger = logging.getLogger(__name__)


class ValidationWarning(UserWarning):
    pass


def _warn(message):
    warnings.warn(message, ValidationWarning)


# Generic validators


def dl_workload_validator(cls, values):
    return values


# Network validators


def network_base_validator(cls, values):
    return values


def training_network_validator(cls, values):
    logger.debug("Validating training network...")
    return values


def inference_network_validator(cls, values):
    logger.debug("Validating inference network...")
    return values


def cask_rollup_network_validator(cls, values):
    return values


# Operator validators
def operator_base_validator(cls, values):
    return values


def gemm_like_validator(cls, values):
    return values


def elementwise_validator(cls, values):
    return values


def convolution_validator(cls, values):
    required_fields = [
        "filter_h",
        "filter_w",
        "stride_h",
        "stride_w",
        "input_channels",
        "output_channels",
    ]

    # Raise a warning if there are missing fields
    if values.get("operator_type") == OperatorType.Convolution:
        missing_fields = [f for f in required_fields if not values.get(f)]
        if missing_fields:
            logger.debug("Missing fields: %s", missing_fields)
            _warn(
                f'Convolution operator is missing the following fields: {", ".join(missing_fields)}'
            )
    return values


def fully_connected_validator(cls, values):
    return values


def batch_norm_validator(cls, values):
    return values
