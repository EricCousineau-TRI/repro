import logging
import uuid
import sys


def get_logger():
    base_name = "example"
    # Use unique name to avoid accumulating handlers into the same logger.
    uid = str(uuid.uuid4())[:8]
    name = f"{base_name}[{uid}]"
    logger = logging.getLogger(name)

    # Configure terse stream handler.
    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Configure verbose file handler.
    file_handler = logging.FileHandler(
        "/tmp/example_log.txt", mode="w"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    file_handler.setLevel(logging.DEBUG)

    # We must set top-level logging level to ensure handlers fire.
    min_level = min(x.level for x in logger.handlers)
    logger.setLevel(min_level)

    return logger


def main():
    for _ in range(2):
        for _ in range(3):
            logger = get_logger()

        logger.debug("debug")
        logger.info("info")
        logger.warning("warn")
        logger.error("error")

        print()


if __name__ == "__main__":
    main()

"""
[ stderr ]
info
warn
error

info
warn
error


[ file: tail -f /tmp/example_log.txt ]

tail: /tmp/example_log.txt: file truncated
2024-10-31 15:53:31,117 - example[029f530a] - DEBUG - debug
2024-10-31 15:53:31,117 - example[029f530a] - INFO - info
2024-10-31 15:53:31,117 - example[029f530a] - WARNING - warn
2024-10-31 15:53:31,117 - example[029f530a] - ERROR - error
tail: /tmp/example_log.txt: file truncated
2024-10-31 15:53:31,118 - example[b45997ba] - DEBUG - debug
2024-10-31 15:53:31,118 - example[b45997ba] - INFO - info
2024-10-31 15:53:31,118 - example[b45997ba] - WARNING - warn
2024-10-31 15:53:31,118 - example[b45997ba] - ERROR - error
"""
