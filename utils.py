import time
import json
import os
import re
import zipfile
import shutil
from PIL import Image
import numpy as np
from typing import (
    List, 
    Optional, 
    Tuple, 
    Iterator,
)


class TimingContextManager:
    def __init__(self, message: str = ""):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        execution_time = end_time - self.start_time
        print(f"{self.message} took {execution_time} seconds to execute.")


def numpy_to_pil_and_save(np_image, output_path):
    # Convert from channel-height-width back to height-width-channel
    np_image = np.transpose(np_image, (1, 2, 0))

    # Denormalize
    np_image = (np_image + 1) * 127.5

    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(np_image.astype("uint8"))

    # Save the PIL image to the specified output path
    pil_image.save(output_path)


def split_list(input_list, chunk_size):
    sublists = []
    for i in range(0, len(input_list), chunk_size):
        sublist = input_list[i : i + chunk_size]
        sublists.append(sublist)
    return sublists


def list_files_in_zip(zip_file_path):
    """
    List the names of all files in a zip archive.

    :param zip_file_path: Path to the zip file.
    :return: A list of file names in the zip archive.
    """
    with zipfile.ZipFile(zip_file_path, "r") as archive:
        file_list = archive.namelist()
    return file_list


def read_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            json_data = file.read()
            data = json.loads(json_data)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None


def save_dict_to_json(dictionary, file_path):
    with open(file_path, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)


def create_abs_path(file_name):
    # Get the directory of the currently running script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the file
    file_path = os.path.join(script_dir, file_name)

    return file_path


def regex_search_list(input_list: List[str], pattern: str) -> List[str]:
    # Compile the regex pattern for efficiency
    compiled_pattern: Pattern[str] = re.compile(pattern)

    # Use list comprehension to filter and collect matching strings
    matched_strings: List[str] = [
        string for string in input_list if compiled_pattern.search(string)
    ]

    return matched_strings


def flatten_list(nested_list) -> list:
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def list_files_in_directory(directory_path):
    try:
        # Get a list of files in the specified directory
        file_list = os.listdir(directory_path)
        return file_list
    except OSError as e:
        # Handle any potential errors, such as the directory not existing
        print(f"An error occurred: {e}")
        return []


def create_batches_from_list(data, batch_size) -> Iterator:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def delete_file_or_folder(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
            print(f"{path} (file) deleted successfully")
        elif os.path.isdir(path):
            try:
                shutil.rmtree(path)
                print(f"{path} (folder) deleted successfully")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"{path} is neither a file nor a folder")
    else:
        print(f"{path} does not exist")


def write_list_to_file(list, output_file):
    try:
        with open(output_file, "w") as file:
            for data in list:
                file.write(data + "\n")
    except Exception as e:
        print(f"Error writing to file: {str(e)}")