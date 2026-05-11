import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Centralized logging configuration.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Reduce noise from 3rd party libs
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("polars").setLevel(logging.WARNING)
