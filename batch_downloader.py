import os
from huggingface_hub import HfFileSystem, hf_hub_url
from typing import Optional, List, Tuple
import random
from huggingface_hub import hf_hub_download
import json
import subprocess
import zipfile
import pandas as pd
from PIL import Image
from threading import Thread
import time
from multiprocessing import Pool, Process
from utils import (
    save_dict_to_json,
    regex_search_list,
    flatten_list,
    list_files_in_directory,
    create_batches_from_list,
    delete_file_or_folder,
)

# TODO: move all helper function into separate module!


def concatenate_csv_files(file_paths: List[str]) -> pd.DataFrame:
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
        thread = Thread(
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
    get_zip_files = regex_search_list(list_files, r"zip")

    # Step 3: Remove '.zip' extension to extract common path patterns
    get_common_path = [f"{x.split('.')[0]}\." for x in get_zip_files]

    # Step 4: Retrieve the list of bundle files based on common path patterns
    get_bundle_files = [regex_search_list(list_files, x) for x in get_common_path]

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


def check_image_error_in_zip(zip_file_path, png_file_name) -> list:
    with zipfile.ZipFile(zip_file_path, "r") as archive:
        with archive.open(png_file_name) as filename:
            list_broken_image = []
            try:
                with Image.open(filename) as im:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
            except Exception as e:
                print(f"image error {png_file_name}: {e}")
                list_broken_image.append(png_file_name)
            return list_broken_image


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
    ramdisk_path = storage_path

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


# deprecated soon
def prefetch_data(
    ramdisk_path: str,
    repo_name: str,
    token: str,
    repo_path: str,
    batch_number: int,
    batch_size: int = 2,
    numb_of_prefetched_batch: int = 1,
    seed: int = 42,
    _batch_name: str = "batch_",
) -> None:
    # prefetch multiple batch in advance to prevent download latency during training
    prefetcher_threads = []

    for thread_count in range(numb_of_prefetched_batch):
        prefetcher_thread = Thread(
            target=download_chunks_of_dataset,
            kwargs={
                "repo_name": repo_name,
                "batch_size": batch_size,
                "offset": batch_size * (batch_number + 1 + thread_count),
                "token": token,
                "repo_path": repo_path,
                "storage_path": ramdisk_path,
                "seed": seed,
                "batch_number": batch_number + 1 + thread_count,
                "batch_name": _batch_name,
                "_temp_file_name": f"{_batch_name}{batch_number+1+thread_count}.txt",
            },
        )

        prefetcher_threads.append(prefetcher_thread)

    # Start the threads
    for thread in prefetcher_threads:
        thread.start()

    # This shouldn't run if the prefetchers succeed in downloading the entire thing and skip to the next line
    download_chunks_of_dataset(
        repo_name=repo_name,
        batch_size=batch_size,
        offset=batch_size * batch_number,
        token=token,
        repo_path=repo_path,
        storage_path=ramdisk_path,
        seed=seed,
        batch_number=batch_number,
        batch_name=_batch_name,
        _temp_file_name=f"{_batch_name}{batch_number}.txt",
    )


def validate_files_in_parallel(
    files_to_check: List[List[str]], numb_of_validator_threads: Optional[int] = 80 * 32
) -> Tuple[List[str], float]:
    """
    Validates files in parallel using multiprocessing.

    Args:
        files_to_check (list): a list containing iterable containing zip file name and file name
            ie: [("zip_file_path1", "filename_in_zip_file1"), ("zip_file_path2", "filename_in_zip_file2")]
        numb_of_validator_threads (int, optional): The number of processes or threads to use.
            Defaults to 80 * 32 (please change this if you're not using TPU lol).

    Returns:
        List[str]: A list of broken file names.
    """
    start = time.time()

    broken_files = []
    # chunk into multiple batches
    for validation_batches in create_batches_from_list(
        files_to_check, numb_of_validator_threads
    ):
        # do parallel validation using multiprocessing
        with Pool(processes=numb_of_validator_threads) as pool:
            results = pool.starmap(check_image_error_in_zip, validation_batches)
        # store the broken file name as string
        broken_files.append(results)

    broken_files = flatten_list(broken_files)
    stop = time.time()
    time_taken = stop - start
    return broken_files, time_taken


def validate_downloaded_batch(
    absolute_batch_path: str,
    prefix: str,
    csv_filenames_col: str,
    numb_of_validator_threads: Optional[int] = 80 * 32,
    _csv_zip_file_path_col: str = "zip_file_path",
    _debug_mode_validation: Optional[bool] = False,
) -> Tuple[List[str], float]:
    """
    Validates files in a downloaded batch in parallel using multiprocessing.

    Args:
        absolute_batch_path (str): The absolute path to the downloaded batch directory.
        prefix (str): Prefix for the zip file paths.
        csv_filenames_col (str): Column name for storing the filenames in the DataFrame.
        numb_of_validator_threads (int, optional): The number of processes or threads to use for validation.
            Defaults to 80 * 32 (please change this if you're not using TPU lol).
        _csv_zip_file_path_col (str): Column name for storing the zip file paths in the DataFrame (default: "zip_file_path").
        _debug_mode_validation (Optional[bool]): only validates a fraction of the files.
    """

    file_list = list_files_in_directory(absolute_batch_path)

    # Get the csvs and convert them to absolute paths
    csvs = regex_search_list(file_list, r".csv")
    csvs = [os.path.join(absolute_batch_path, csv) for csv in csvs]

    # Get the zips and convert them to absolute paths
    zips = regex_search_list(file_list, r".zip")
    zips = [os.path.join(absolute_batch_path, zip_file) for zip_file in zips]

    # Combine csvs into one dataframe
    df_caption = concatenate_csv_files(csvs)

    # Create zip file path for each image to indicate where the image resides inside the zip
    df_caption[_csv_zip_file_path_col] = (
        absolute_batch_path + "/" + prefix + df_caption.chunk_id + ".zip"
    )

    # Store filename and zip folder in a list
    # [(file1, zip_path1), (file2, zip_path2)]
    file_to_check = list(
        zip(
            df_caption[_csv_zip_file_path_col].tolist(),
            df_caption[csv_filenames_col].tolist(),
        )
    )
    if _debug_mode_validation and len(file_to_check) > numb_of_validator_threads:
        print(
            f"debug mode: only checking {numb_of_validator_threads} files out of {len(file_to_check)}"
        )
        file_to_check = file_to_check[:numb_of_validator_threads]

    broken_files, time_taken = validate_files_in_parallel(
        files_to_check=file_to_check,
        numb_of_validator_threads=numb_of_validator_threads,
    )
    return broken_files, time_taken


def unpack_zip_files(absolute_batch_path: str, absolute_target_dir: str) -> None:
    """
    unzip the image file into one directory

    Args:
        absolute_batch_path (str): The absolute path to the downloaded batch directory.
        absolute_target_dir (str): The target directory where the files will be extracted.
    """

    file_list = list_files_in_directory(absolute_batch_path)

    # Get the zips and convert them to absolute paths
    zips = regex_search_list(file_list, r".zip")
    zips = [os.path.join(absolute_batch_path, zip_file) for zip_file in zips]

    # gonna use lambda function, it's shorter :P
    unzip_file = lambda zip_path, target_dir: subprocess.run(
        ["7z", "x", zip_path, f"-o{target_dir}", "-Y"]
    )

    threads = []

    for zip_file in zips:
        thread = Thread(target=unzip_file, args=(zip_file, absolute_target_dir))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def download_chunks_of_dataset_with_validation(
    repo_name: str,
    batch_size: int,
    offset: int,
    storage_path: str,
    batch_number: str,
    prefix: str,
    numb_of_validator_threads: Optional[int] = 80 * 32,
    batch_name: Optional[str] = "batch_",
    token: Optional[str] = None,
    repo_path: Optional[str] = None,
    seed: Optional[int] = 42,
    _csv_zip_file_path_col: str = "zip_file_path",
    _temp_file_name: Optional[str] = "aria_download_url_temp.txt",
    _manifest_file_name: Optional[str] = "manifest.json",
    _image_folder_name: Optional[str] = "image",
    _debug_mode_validation: Optional[bool] = False,
    _disable_validation: Optional[bool] = False,
) -> None:
    """
    Download data chunks from a specified repository using the Aria2 download manager.

    Args:
        repo_name (str): The name of the repository to download data from.
        batch_size (int): The number of items to download in each batch.
        offset (int): The starting index of the dataset to download.
        storage_path (str): The directory where downloaded data will be stored.
        batch_number (str): A unique identifier for the current download batch.
        prefix (str): Prefix for the zip file paths.
        _csv_zip_file_path_col (str): Column name for storing the zip file paths in the DataFrame (default: "zip_file_path").
        numb_of_validator_threads (int, optional): The number of processes or threads to use for validation.
            Defaults to 80 * 32 (not implemented yet!).
        batch_name (Optional[str]): Prefix for batch directory names (default: "batch_").
        token (Optional[str]): Authentication token if required (default: None).
        repo_path (Optional[str]): The path to the specific dataset within the repository (default: None).
        seed (Optional[int]): Random seed for data sampling (default: 42).
        _temp_file_name (Optional[str]): Temporary file name for storing download URLs (default: "aria_download_url_temp.txt").
        _manifest_file_name (Optional[str]): manifest file that contains batch details (default: ""manifest.json").
        _debug_mode_validation (Optional[bool]): only validates a fraction of the files.
        _disable_validation (Optional[bool]): disable validation mode entirely.
    """
    # convert to absolute path
    ramdisk_path = storage_path
    download_dir = os.path.join(ramdisk_path, f"{batch_name}{batch_number}")
    urls_file = os.path.join(ramdisk_path, _temp_file_name)
    manifest_file = os.path.join(download_dir, "manifest.json")

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
    write_urls_to_file(aria_format, urls_file)

    if not os.path.exists(manifest_file):
        # use aria to download everything
        download_with_aria2(
            download_directory=download_dir,
            urls_file=urls_file,
            auth_token=token,
        )

        # unzip and put everything into one folder
        unpack_zip_files(
            absolute_batch_path=download_dir,
            absolute_target_dir=os.path.join(download_dir, _image_folder_name),
        )
        # delete the zip
        file_list = list_files_in_directory(download_dir)
        zips = regex_search_list(file_list, r".zip")
        zips = [os.path.join(download_dir, zip_file) for zip_file in zips]
        for zip_file in zips:
            delete_file_or_folder(zip_file)

        print(f"creating manifest file for batch {batch_name}{batch_number}")

        # just store this details for now
        manifest = {"image_folder": _image_folder_name}
        print(
            f"manifest file for batch {batch_name}{batch_number} created and stored at {manifest_file}"
        )
        save_dict_to_json(manifest, manifest_file)

    else:
        print(
            f"manifest file for this {batch_name}{batch_number} exist, skipping download for this batch"
        )

    if _disable_validation:
        NotImplementedError

    if _debug_mode_validation:
        NotImplementedError


def prefetch_data_with_validation(
    ramdisk_path: str,
    repo_name: str,
    token: str,
    repo_path: str,
    batch_number: int,
    prefix: str,
    numb_of_validator_threads: Optional[int] = 80 * 32,
    batch_size: int = 2,
    numb_of_prefetched_batch: int = 1,
    seed: int = 42,
    _csv_zip_file_path_col: str = "zip_file_path",
    _batch_name: str = None,
    _debug_mode_validation: Optional[bool] = False,
    _disable_validation: Optional[bool] = False,
) -> None:
    """
    Prefetch data with validation from a remote repository into a local storage.

    Args:
        ramdisk_path (str): The path to the local storage (RAM disk) where data will be stored.
        repo_name (str): The name of the remote repository.
        token (str): The authentication token for accessing the remote repository.
        repo_path (str): The path within the remote repository where data is located.
        batch_number (int): The batch number to process.
        prefix (str): Prefix for the zip file paths.
        numb_of_validator_threads (int, optional): The number of processes or threads to use for validation.
            Defaults to 80 * 32 (not implemented yet!).
        batch_size (int, optional): The batch size. Defaults to 2.
        numb_of_prefetched_batch (int, optional): The number of batches to prefetch in advance. Defaults to 1.
        seed (int, optional): The random seed for data retrieval. Defaults to 42.
        _csv_zip_file_path_col (str): Column name for storing the zip file paths in the DataFrame Defaults to "zip_file_path".
        _batch_name (str, optional): The base name for the batches. Defaults to `prefix`.
        _debug_mode_validation (Optional[bool]): only validates a fraction of the files.
        _disable_validation (Optional[bool]): disable validation mode entirely.
    """
    if _batch_name == None:
        _batch_name = prefix
    # prefetch multiple batch in advance to prevent download latency during training
    prefetcher_processes = []

    for thread_count in range(numb_of_prefetched_batch):
        prefetcher_thread = Process(
            target=download_chunks_of_dataset_with_validation,
            kwargs={
                "repo_name": repo_name,
                "batch_size": batch_size,
                "offset": batch_size * (batch_number + 1 + thread_count),
                "token": token,
                "repo_path": repo_path,
                "storage_path": ramdisk_path,
                "seed": seed,
                "batch_number": batch_number + 1 + thread_count,
                "batch_name": _batch_name,
                "prefix": prefix,
                "_csv_zip_file_path_col": _csv_zip_file_path_col,
                "numb_of_validator_threads": numb_of_validator_threads,
                "_debug_mode_validation": _debug_mode_validation,
                "_temp_file_name": f"{prefix}{batch_number+1+thread_count}.txt",
                "_disable_validation": _disable_validation,
            },
        )

        prefetcher_processes.append(prefetcher_thread)

    # Start the threads
    for process in prefetcher_processes:
        process.start()

    # This shouldn't run if the prefetchers succeed in downloading the entire thing and skip to the next line
    download_chunks_of_dataset_with_validation(
        repo_name=repo_name,
        batch_size=batch_size,
        offset=batch_size * batch_number,
        token=token,
        repo_path=repo_path,
        storage_path=ramdisk_path,
        seed=seed,
        batch_number=batch_number,
        batch_name=_batch_name,
        prefix=prefix,
        _csv_zip_file_path_col=_csv_zip_file_path_col,
        numb_of_validator_threads=numb_of_validator_threads,
        _debug_mode_validation=_debug_mode_validation,
        _temp_file_name=f"{prefix}{batch_number}.txt",
        _disable_validation=_disable_validation,
    )
