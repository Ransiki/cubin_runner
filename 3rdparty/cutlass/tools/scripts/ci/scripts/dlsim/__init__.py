import sys
import os
from pathlib import Path
from enum import Enum

sys.path.append(os.path.join(
    Path(os.path.realpath(__file__)).parent.parent.parent.parent.parent.parent,
    "bloom", "testing"    
))

class PerfList(Enum):
    perf_smart = 'perf_smart'
    perf_perfsim = 'perf_perfsim'

    def __str__(self):
        return self.value
