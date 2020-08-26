import os
import logging
import shlex
from subprocess import check_output
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

repo_root = check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii')
log_file = os.path.join(repo_root, 'output/%s/log' % timestamp)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()