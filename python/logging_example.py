import logging
import uuid
import sys


def get_logger():
    uid = str(uuid.uuid4())[:8]
    logger = logging.getLogger(f"example[{uid}]")
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
Currently, doesn't respect level?

console:

warn
error

warn
error


file:

tail: /tmp/example_log.txt: file truncated
2024-10-31 15:47:39,300 - example[3b20f328] - WARNING - warn
2024-10-31 15:47:39,300 - example[3b20f328] - ERROR - error
tail: /tmp/example_log.txt: file truncated
2024-10-31 15:47:39,300 - example[7ca514ac] - WARNING - warn
2024-10-31 15:47:39,300 - example[7ca514ac] - ERROR - error
"""
