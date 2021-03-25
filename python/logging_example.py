#!/usr/bin/env python3

import logging


def main():
    log = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
    )
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.info("Hey")


if __name__ == "__main__":
    main()
