import json
import logging


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    formatter = logging.Formatter("{asctime} - {levelname} - {message}", style="{", datefmt="%Y-%m-%d %H:%M")
    console_handler.setFormatter(formatter)

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
