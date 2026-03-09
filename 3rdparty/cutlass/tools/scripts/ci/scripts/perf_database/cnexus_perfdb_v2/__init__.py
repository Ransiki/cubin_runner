import sys
import logging

from .api import Api, Env  # noqa
from .models import *      # noqa

if sys.version_info < (3, 6, 0):
    raise RuntimeError("Python version >=3.6 required")

# Reduce logging a bit
for module in ("boto", "botocore", "s3transfer", "urllib3"):
    logging.getLogger(module).setLevel(logging.ERROR)
