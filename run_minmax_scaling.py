from src import logging
from src.data.preprocessing import (find_target_ghi_max_value,
                                    find_target_ghi_min_value)

logger = logging.create_logger(__name__)


def main():
    max_target = find_target_ghi_max_value()
    min_target = find_target_ghi_min_value()

    logger.info(f"Max target value: {max_target}")
    logger.info(f"Min target value: {min_target}")


if __name__ == "__main__":
    main()
