# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.51"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    # default for reduced CPU utilization during training
    os.environ["OMP_NUM_THREADS"] = "1"

from ultralytics.models import YOLO
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "checks",
    "download",
    "settings",
)
