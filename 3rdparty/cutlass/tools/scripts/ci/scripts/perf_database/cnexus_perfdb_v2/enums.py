from .base import LooseEnum as Enum
from enum import EnumMeta
import re

__all__ = ["MetricType"]

MULTI_SYSTEM_COMPONENTS = {
    "gpu": {
        "clock",
        "util",
        "temp",
        "fan_speed",
        "graphics_clock",
        "power_draw",
        "memory_clock",
        "memory_util",
        "memory_temp",
        "decoder_util",
        "encoder_util",
    },
    "cpu": {"clock", "util", "temp", "power_draw", "fan_speed"},
}

SUFFIX_PERMUTATIONS = {"cpu": {"core"}, "gpu": {}}


class HardwarePermutationMetaClass(EnumMeta):
    def __getattr__(cls, name):
        """Return the enum member matching `name`

        We use __getattr__ instead of descriptors or inserting into the enum
        class' __dict__ in order to support `name` and `value` being both
        properties for enum members (which live in the class' __dict__) and
        enum members themselves.

        This was added to allow clients to access fields on the `MetricType`
        class as: `MetricType.cpu_{core info}_{cpu component}` without
        explicitly adding them to the Enum, as there could be a variable number
        of cores within a system.
        """
        name = name.lower()
        for component in MULTI_SYSTEM_COMPONENTS.keys():
            search = re.search(
                rf"^{component}_(?:([a-zA-Z]+)_?(\d+))?_([a-zA-Z]+)$", name
            )
            if search:
                subcomponent, str_subcomp_index, measurement = search.groups()
                final_subcomponent = ""
                if subcomponent and subcomponent in SUFFIX_PERMUTATIONS[component]:
                    final_subcomponent = f"{subcomponent}"
                    if str_subcomp_index:
                        final_subcomponent += f"_{int(str_subcomp_index)}"

                if measurement in MULTI_SYSTEM_COMPONENTS[component]:
                    return f"{component}_{final_subcomponent}_{measurement}".lower()
                break

        return EnumMeta.__getattr__(cls, name)


class MetricType(Enum, metaclass=HardwarePermutationMetaClass):
    """
    An enum storing all possible measurement values to be stored in the db.
    Note all index numbers are 0 based.

    Example Usages:

    - `cpu_util_core_1`: CPU core 1 % utilization
    - `cpu_util`: CPU total utilization %.
    - `cpu_core_0_util`: CPU core 0 total utilization %.
    - `cpu_core1_util`: CPU core 1 total utilization %.

    Linters may flag cpu core permutations as they're dynamically added.

    Tegra statistics source data:
    - https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/AppendixTegraStats.html
    - https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_nano.html
    """

    # GPU Metric Types
    gpu_clock = "gpu_clock"  #: Clock speed of GPU in MHz
    gpu_decoder_util = "gpu_decoder_util"  #: GPU decoder utilization as %
    gpu_encoder_util = "gpu_encoder_util"  #: GPU encoder utilization as %
    gpu_fan_speed = "gpu_fan_speed"  #: % Utilization of fan in GPU
    gpu_graphics_clock = "gpu_graphics_clock"  #: GPU graphics clock speed in MHz
    gpu_memory_clock = "gpu_memory_clock"  #: Clock speed of the GPU memory in MHz
    gpu_memory_temp = "gpu_memory_temp"  #: GPU memory temperature as C
    gpu_memory_util = "gpu_memory_util"  #: GPU memory util as %
    gpu_perf_state = "gpu_perf_state"  #: Numeric status code of the GPU
    gpu_power_draw = "gpu_power_draw"  #: Power draw of GPU in W
    gpu_process_num = "gpu_process_num"  #: Number of processes using GPU resources
    gpu_temp = "gpu_temp"  #: Temperature of GPU in C
    gpu_util = "gpu_util"  #: Utilization % of GPU

    # CPU Metric Types
    cpu_clock = "cpu_clock"  #: Clock speed of cpu in MHz
    cpu_fan_speed = "cpu_fan_speed"  #: % Utilization of cpu fan
    cpu_power_draw = "cpu_power_draw"  #: Power draw of the cpu in W
    cpu_temp = "cpu_temp"  #: Temperature of cpu in C
    cpu_util = "cpu_util"  #: Utilization % of cpu

    # System Metric Types
    motherboard_temp = "motherboard_temp"  #: Motherboard temperature (or circuit board temp for tegra systems) in C
    system_bg_util = "system_bg_util"  #: Total time spent on background tasks as a %
    system_fg_util = "system_fg_util"  #: Total time spent on foreground tasks as a %
    system_power_draw = "system_power_draw"  #: Total power draw of the system in W

    # Misc util/usages
    ram_usage = "ram_usage"  #: Total RAM usage of the system as MB
    ram_util = "ram_util"  #: Total RAM usage of the system as %
    swap_usage = "swap_usage"  #: Total SWAP usage of the system as MB
    swap_util = "swap_util"  #: Total SWAP usage of the system as %

    # Tegra register specific metrics
    ao_temp = "ao_temp"  #: "Always-On" register temperature value in C
    ape_freq = "ape_freq"  #: Audio processing engine frequency in MHz
    aux_temp = (
        "aux_temp"  #: Temperature of on-chip ring oscillators and CV cluster as C
    )
    cv_power_draw = "cv_power_draw"  #: CV power rail cluster power draw in W
    emc_freq = "emc_freq"  #: Total external memory controller freq as MHz
    emc_util = "emc_util"  #: Total external memory controller util as a %
    pmic_temp = (
        "pmic_temp"  #: Temperature of power management integrated controller as C
    )
    soc_power_draw = "soc_power_draw"  #: SOC power rail cluster power draw in W
    sys5v_power_draw = "sys5v_power_draw"  #: SYS5V power rail power consumption in W
    vddrq_power_draw = "vddrq_power_draw"  #: VDDRQ power rail power consumption in W
    vic_freq = "vic_freq"  #: Total video image compositor freq as MHz
    vic_util = "vic_util"  #: Total video image compositor util as a %

    tegra_fanctl_est_temp = "tegra_fanctl_est_temp"  #: The full name is thermal-fan-est.  It is a weighted average of soctherm CPU,GPU..etc thermal sensor which is used to control fan speed. Unit is C.
    tegra_board_temp = "tegra_board_temp"  #: Module Temperature measured by external thermal sensor (TMP451).  It can be used for skin-temperature calculation but mostly not used in dev-kits as it a open platform. Unit is C.
