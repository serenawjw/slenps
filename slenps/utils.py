import sys
import psutil
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")


def check_memory_usage(*objs, warning_percent=25):
    """
    Check the combined memory usage of multiple objects and log warnings based on system memory constraints.

    Args:
    *objs: Variable length argument list of objects to check memory usage for.
    """
    # Calculate the combined size of the objects in gigabytes
    total_obj_size_gb = sum(sys.getsizeof(obj) / (1024**3) for obj in objs)

    # Retrieve system memory details
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024**3)
    total_memory_gb = memory.total / (1024**3)

    # Calculate the percent of total memory that the objects occupy
    memory_percent = (total_obj_size_gb / total_memory_gb) * 100

    if memory_percent > warning_percent:
        logging.warning(
            f"The objects occupy more than 25% of the total memory: {memory_percent:.2f}%"
        )

    # Log a warning if the current memory usage is less than twice the size of the objects
    if available_memory_gb < 2 * total_obj_size_gb:
        logging.warning(
            f"available memory usage is less than twice the size of the objects. "
            + f"Current available memory: {available_memory_gb:.2f} GB, Objects size: {total_obj_size_gb:.2f} GB"
        )


# Example of usage:
# embeddings, documents = get_data_from_paths(filepaths)
# check_memory_usage(embeddings, documents)
