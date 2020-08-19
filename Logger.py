import logging
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler('output/log/%s.log' % timestamp),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()