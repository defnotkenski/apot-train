import json
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme


def setup_logger() -> logging.Logger:
    # Set up the logger with pertyyy Rich logging

    rich_console = Console(stderr=True)

    rh = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        log_time_format="%H:%M:%S-%f",
        level=logging.DEBUG,
        console=rich_console,
    )

    logger = logging.getLogger(__name__)
    logger.addHandler(rh)

    return logger


def sort_json(json_path, output_name):
    # Sort a JSON file containing key, value pairs based on keys

    with open(f"{json_path}", "r") as json_read, open(f"{output_name}.json", "w") as json_write:
        json_dict = json.load(json_read)
        json.dump({k: json_dict[k] for k in sorted(json_dict)}, json_write, indent=2)

    return f"Saved file as {output_name}.json"


if __name__ == "__main__":
    # Remember to change path

    # print(sort_json("config_dreambooth.json", "config_dreambooth"))

    test_logger = setup_logger()
    test_logger.warning("Ayyye")
