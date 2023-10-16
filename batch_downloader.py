import os
from huggingface_hub import HfFileSystem, hf_hub_url
from typing import Optional, List, Pattern, Tuple
import re
import random
from huggingface_hub import hf_hub_download
import threading
import json
import subprocess
import zipfile
import pandas as pd
from PIL import Image


def concatenate_csv_files(file_paths:List[str]) -> pd.DataFrame:
    try:
        # Initialize an empty DataFrame to store the concatenated data
        concatenated_data = pd.DataFrame()

        # Loop through the list of file paths and concatenate the CSV files
        for file_path in file_paths:
            # Read each CSV file into a DataFrame
            data = pd.read_csv(file_path)
            
            # Concatenate the data to the existing DataFrame
            concatenated_data = pd.concat([concatenated_data, data], ignore_index=True)

        return concatenated_data

    except Exception as e:
        # Handle any potential errors
        print(f"An error occurred: {e}")


def list_files_in_zip(zip_file_path):
    """
    List the names of all files in a zip archive.

    :param zip_file_path: Path to the zip file.
    :return: A list of file names in the zip archive.
    """
    with zipfile.ZipFile(zip_file_path, "r") as archive:
        file_list = archive.namelist()
    return file_list


def download_with_aria2(download_directory, urls_file, auth_token):
    # Build the command as a list of arguments
    command = [
        "aria2c",
        "--input-file",
        urls_file,
        "--dir",
        download_directory,
        "--max-concurrent-downloads",
        "48",
        "--max-connection-per-server",
        "16",
        "--log",
        "aria2.log",
        "--log-level=error",
        "--auto-file-renaming=false",
        "--max-tries=3",
        "--retry-wait=5",
        "--user-agent=aria2c/pow",
        "--header",
        f"Authorization: Bearer {auth_token}",  # Replace 'Bearer' with your token type if needed
    ]

    try:
        # Execute the aria2c command
        subprocess.run(command, check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Download failed with error: {e.returncode}")


def write_urls_to_file(url_list, output_file):
    try:
        with open(output_file, "w") as file:
            for url in url_list:
                file.write(url + "\n")
        print(f"URLs have been written to {output_file}")
    except Exception as e:
        print(f"Error writing URLs to file: {str(e)}")


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


def create_abs_path(file_name):
    # Get the directory of the currently running script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the file
    file_path = os.path.join(script_dir, file_name)

    return file_path


def download_files_concurently(
    list_of_hf_dir: list,
    repo_name: str,
    token: str,
    repo_type: str,
    download_path_dir: str,
) -> None:
    """
    Downloads files concurrently from Hugging Face Hub using multithreading.

    Args:
        list_of_hf_dir (list): A list of filenames to be downloaded concurrently.
        repo_name (str): The name of the Hugging Face repository to download from.
        token (str): The Hugging Face API token for authentication (or None if not needed).
        repo_type (str): The type of repository (e.g., 'model', 'dataset', 'script').
        download_path_dir (str): The local directory where downloaded files will be saved.

    Returns:
        None

    This function initiates multiple threads to download files from the Hugging Face Hub concurrently.
    Each thread is responsible for downloading a specific file and saving it to the local directory.

    Example:
    ```
    list_of_files = ["file1.pth", "file2.pth"]
    repo_name = "username/repo-name"
    token = "your_api_token"
    repo_type = "model"
    download_path_dir = "/path/to/save/files"
    download_files_concurrently(list_of_files, repo_name, token, repo_type, download_path_dir)
    ```

    Note:
    - Make sure to provide a valid Hugging Face API token if required.
    - Ensure that the `list_of_hf_dir` contains the correct filenames for the repository.
    """
    threads = []
    for filename in list_of_hf_dir:
        thread = threading.Thread(
            target=hf_hub_download,
            kwargs={
                "repo_id": repo_name,
                "filename": filename,
                "token": token,
                "repo_type": repo_type,
                "local_dir": download_path_dir,
                "local_dir_use_symlinks": False,
            },
        )
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def get_list_of_files_from_hf(
    repo_name: str, token: Optional[str] = None, repo_path: Optional[str] = None
) -> List[str]:
    """
    Get a list of files from a Hugging Face dataset repository.

    This function allows you to retrieve a list of file paths from a Hugging Face dataset repository on the Hugging Face Hub.

    Args:
        repo_name (str): The name of the Hugging Face repository (e.g., "username/dataset-name").
        token (str, optional): Your Hugging Face API token. If provided, allows access to private repositories. Default is None.
        repo_path (str, optional): The subfolder path within the repository to navigate to. Default is None.

    Returns:
        List[str]: A list of file paths within the specified Hugging Face dataset repository or subfolder.

    Example:
        >>> repo_name = "username/dataset-name"
        >>> token = "YOUR_API_TOKEN"
        >>> repo_path = "subfolder"  # Optional, if you want to navigate to a subfolder
        >>> file_paths = get_list_of_files_from_hf(repo_name, token, repo_path)
        >>> print(file_paths)
        ['file1.txt', 'file2.csv', 'subfolder/file3.json']

    Note:
        - To use this function, you need to have the `huggingface_hub` library installed.
        - You can obtain your Hugging Face API token from https://huggingface.co/settings/token.
        - If no `repo_path` is provided, the function retrieves files from the root of the repository.

    """
    # Construct the dataset path including the optional subfolder
    if repo_path:
        dataset_path = f"datasets/{repo_name}/{repo_path}"
    else:
        dataset_path = f"datasets/{repo_name}"

    # Initialize the Hugging Face FileSystem with the provided token
    fs = HfFileSystem(token=token)

    # Get a list of all files inside the specified directory
    file_paths = fs.ls(dataset_path, detail=False)

    return file_paths


def regex_search_list(input_list: List[str], pattern: str) -> List[str]:
    # Compile the regex pattern for efficiency
    compiled_pattern: Pattern[str] = re.compile(pattern)

    # Use list comprehension to filter and collect matching strings
    matched_strings: List[str] = [
        string for string in input_list if compiled_pattern.search(string)
    ]

    return matched_strings


def convert_filenames_to_urls(repo_name: str, file_names: List[str]) -> List[str]:
    # Generate the URL for each file using Hugging Face Hub's `hf_hub_url` function
    urls = [
        hf_hub_url(repo_name, filename, repo_type="dataset") for filename in file_names
    ]

    return urls


def get_generate_urls_and_file_name(
    repo_name: str,
    token: Optional[str] = None,
    repo_path: Optional[str] = None,
    seed: int = 42,
) -> Tuple[List[str]]:
    """
    Get shuffled URLs of bundle files from a Hugging Face dataset repository.
    """
    # Step 1: Get a list of files from the Hugging Face repository or subfolder
    list_files = get_list_of_files_from_hf(
        token=token, repo_name=repo_name, repo_path=repo_path
    )

    # Step 2: Find zip file names to retrieve corresponding CSV files
    get_zip_files = regex_search_list(list_files, r".zip")

    # Step 3: Remove '.zip' extension to extract common path patterns
    get_common_path = [x.split(".")[0] for x in get_zip_files]

    # Step 4: Retrieve the list of bundle files based on common path patterns
    get_bundle_files = [regex_search_list(list_files, x)[1:] for x in get_common_path]

    # Step 5: remove huggingface repository path
    get_bundle_files = [
        ["/".join(x.split("/")[3:]) for x in bundle] for bundle in get_bundle_files
    ]
    # print(get_bundle_files)

    # Step 6: Create a shuffle order
    order = list(range(len(get_bundle_files)))

    # Step 7: Shuffle the order based on the provided seed
    random.seed(seed)
    shuffled_list = random.sample(order, len(order))

    # Step 8: Convert bundle file names to their corresponding URLs
    shuffled_urls = [
        convert_filenames_to_urls(repo_name=repo_name, file_names=x)
        for x in get_bundle_files
    ]

    return [shuffled_urls[i] for i in shuffled_list], [
        get_bundle_files[i] for i in shuffled_list
    ]


def get_sample_from_repo(
    repo_name: str,
    batch_size: int,
    token: Optional[str] = None,
    repo_path: Optional[str] = None,
    seed: Optional[int] = 42,
    offset: Optional[int] = 0,
) -> Tuple[List[str]]:
    """
    Get sampled URLs of bundle files from a Hugging Face dataset repository.
    """
    url_and_file_name = get_generate_urls_and_file_name(
        repo_name=repo_name,
        token=token,
        repo_path=repo_path,
        seed=seed,
    )

    # store batch here
    url_batches = []
    file_name_batches = []

    counter = 0
    while counter != batch_size:
        # cycle around the download url
        modulo_batches = (counter + offset) % len(url_and_file_name[0])
        counter += 1
        # store link to be downloaded
        url_batches.append(url_and_file_name[0][modulo_batches])
        # store file path to be downloaded
        file_name_batches.append(url_and_file_name[1][modulo_batches])
    return url_batches, file_name_batches


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

def check_error(filename: str) -> list:
    list_broken_image = []
    try:
        with Image.open(filename) as im:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
    except Exception as e:
        print(f"image error {filename}: {e}")
        list_broken_image.append(filename)
    return list_broken_image


def process_image_in_zip(zip_file_path, png_file_name, process_func):
    """
    Open an image from a zip file, apply a processing function, and return the result.

    :param zip_file_path: Path to the zip file.
    :param png_file_name: Name of the PNG file inside the zip.
    :param process_func: Function to process the opened image.
    :return: The result of processing the image.
    """
    with zipfile.ZipFile(zip_file_path, "r") as archive:
        with archive.open(png_file_name) as file_in_zip:
            result = process_func(file_in_zip)
    return result


def download_chunks_of_dataset(
    repo_name: str,
    batch_size: int,
    offset: int,
    storage_path: str,
    batch_number: str,
    batch_name: Optional[str] = "batch_",
    token: Optional[str] = None,
    repo_path: Optional[str] = None,
    seed: Optional[int] = 42,
    _temp_file_name: Optional[str] = "aria_download_url_temp.txt",
) -> None:
    """
    Download data chunks from a specified repository using the Aria2 download manager.

    Args:
        repo_name (str): The name of the repository to download data from.
        batch_size (int): The number of items to download in each batch.
        offset (int): The starting index of the dataset to download.
        storage_path (str): The directory where downloaded data will be stored.
        batch_number (str): A unique identifier for the current download batch.
        batch_name (Optional[str]): Prefix for batch directory names (default: "batch_").
        token (Optional[str]): Authentication token if required (default: None).
        repo_path (Optional[str]): The path to the specific dataset within the repository (default: None).
        seed (Optional[int]): Random seed for data sampling (default: 42).
        _temp_file_name (Optional[str]): Temporary file name for storing download URLs (default: "aria_download_url_temp.txt").
    """
    # convert to absolute path
    ramdisk_path = create_abs_path(storage_path)

    data = get_sample_from_repo(
        repo_name=repo_name,
        token=token,
        repo_path=repo_path,
        seed=seed,
        batch_size=batch_size,
        offset=offset,
    )

    # grab the urls and extract file name from urls
    file_urls = flatten_list(data[0])
    # create a list of url strings and args that supported by aria2
    aria_format = [
        f"{file_name}\n\tout={file_name.split('/')[-1]}\n" for file_name in file_urls
    ]
    # put the urls into a temporary txt file so aria can download it
    write_urls_to_file(aria_format, os.path.join(ramdisk_path, _temp_file_name))

    # use aria to download everything
    download_with_aria2(
        download_directory=os.path.join(ramdisk_path, f"{batch_name}{batch_number}"),
        urls_file=os.path.join(ramdisk_path, _temp_file_name),
        auth_token=token,
    )

def delete_file_or_folder(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
            print(f"{path} (file) deleted successfully")
        elif os.path.isdir(path):
            os.rmdir(path)
            print(f"{path} (folder) deleted successfully")
        else:
            print(f"{path} is neither a file nor a folder")
    else:
        print(f"{path} does not exist")
        
# def main():
#     creds_data = "repo.json"
#     ramdisk_path = "ramdisk"
#     temp_file_urls = "download.txt"
#     batch_name = "batch_"
#     batch_number = 1
#     seed = 432
#     prefix = "16384-e6-"

#     # grab token and repo id from json file
#     repo_id = read_json_file(create_abs_path(creds_data))

#     # download batch
#     download_chunks_of_dataset(
#         repo_name=repo_id["repo_name"],
#         batch_size=2,
#         offset=3,
#         token=repo_id["token"],
#         repo_path="chunks",
#         storage_path=ramdisk_path,
#         seed=seed,
#         batch_number=batch_number,
#         batch_name=batch_name
#     )

#     # accessing images and csv data

#     # get list of files in the batch folder
#     batch_path = os.path.join(create_abs_path(ramdisk_path),f"{batch_name}{batch_number}")
#     file_list = list_files_in_directory(batch_path)
#     # get the csvs and convert it to abs path
#     csvs = regex_search_list(file_list, r".csv")
#     csvs = [os.path.join(batch_path, csv) for csv in csvs]
#     # get the zip and convert it to abs path
#     zips = regex_search_list(file_list, r".zip")
#     zips = [os.path.join(batch_path, zip) for zip in zips]

#     # combine csvs into 1 dataframe
#     df_caption = concatenate_csv_files(csvs)
#     # create zip file path for each image to indicate where the image resides inside the zip
#     df_caption["zip_file_path"] = (batch_path+"/"+prefix+df_caption.chunk_id+".zip")
#     # TODO: i think unzipping the files inside a folder is the best choice ? so we dont have to modify the dataloader
#     # alternatively modify the dataloader and not unzip the files

#     list_files_in_zip("ramdisk/batch_1/16384-e6-6c384a84-b69d-4ed7-bde1-cddcea3009d0.zip")
#     print()


# main()
