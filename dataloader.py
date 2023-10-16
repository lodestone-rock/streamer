# python
import os
from threading import Thread
import pandas as pd

from batch_downloader import (
    read_json_file,
    download_chunks_of_dataset,
    regex_search_list,
    concatenate_csv_files,
    create_abs_path,
    list_files_in_directory,
    delete_file_or_folder,
)

from dataframe_processor import create_tag_based_training_dataframe

from batch_processor import (
    generate_batch,
    process_image,
    tokenize_text,
)


# TODO: should i put stuff in class or i can write a neat function instead
class DataLoader:
    def __init__(
        self,
        disk_path: str,
        numb_of_chunk_per_batch: int = 2,
        seed: int = 42,
        temp_file_urls: str = "temp_download_url.txt",
    ) -> None:
        self.numb_of_chunk_per_batch = numb_of_chunk_per_batch


def main():
    creds_data = "repo.json"
    ramdisk_path = "ramdisk"
    temp_file_urls = "download.txt"
    batch_name = "batch_"
    batch_number = 0  # <<< this should be incremented for each sucessfull dataloading
    batch_size = 2
    seed = 432  # <<< this should be incremented when all batch is processed
    prefix = "16384-e6-"
    MAXIMUM_RESOLUTION_AREA = [576**2, 704**2, 832**2, 960**2, 1088**2]
    BUCKET_LOWER_BOUND_RESOLUTION = [384, 512, 576, 704, 832]

    # grab token and repo id from json file
    repo_id = read_json_file(create_abs_path(creds_data))

    # TODO:this download should run in a separate thread and have batch offset
    # no! actually run this function twice, first sequential then concurrent,
    # it wont download the current batch if the data already there. so it will "skips" to the dataloader part
    # after dataloader part is finished and the training loop finished just delete the current batch and increment
    # when it's incremented the download_chunks_of_dataset will try to download next batch but it's already there
    # so it skips again and the thread prefetch the next batch
    # download batch

    # the prefetcher thread
    # the prefetcher ensures the next batch will be ready before the training loop even start
    prefetcher_thread = Thread(
        target=download_chunks_of_dataset,
        kwargs={
            "repo_name": repo_id["repo_name"],
            "batch_size": batch_size,
            "offset": batch_size
            * (
                batch_number + 1
            ),  # prevent batch overlap so it retreive full chunk + 1 for prefetch
            "token": repo_id["token"],
            "repo_path": "chunks",
            "storage_path": ramdisk_path,
            "seed": seed,
            "batch_number": batch_number + 1,
            "batch_name": batch_name,
            "_temp_file_name": f"{batch_name}{batch_number+1}.txt",
        },
    )
    # fork and start thread
    prefetcher_thread.start()

    # this souldn't run if the prefetcher succeed downloading the entire thing and skips to the next line
    download_chunks_of_dataset(
        repo_name=repo_id["repo_name"],
        batch_size=batch_size,
        offset=batch_size
        * batch_number,  # prevent batch overlap so it retreive full chunk
        token=repo_id["token"],
        repo_path="chunks",
        storage_path=ramdisk_path,
        seed=seed,
        batch_number=batch_number,
        batch_name=batch_name,
        _temp_file_name=f"{batch_name}{batch_number}.txt",
    )

    # dataloader part
    # accessing images and csv data
    # get list of files in the batch folder
    batch_path = os.path.join(
        create_abs_path(ramdisk_path), f"{batch_name}{batch_number}"
    )
    file_list = list_files_in_directory(batch_path)
    # get the csvs and convert it to abs path
    csvs = regex_search_list(file_list, r".csv")
    csvs = [os.path.join(batch_path, csv) for csv in csvs]
    # get the zip and convert it to abs path
    zips = regex_search_list(file_list, r".zip")
    zips = [os.path.join(batch_path, zip) for zip in zips]

    # combine csvs into 1 dataframe
    df_caption = concatenate_csv_files(csvs)
    # create zip file path for each image to indicate where the image resides inside the zip
    df_caption["zip_file_path"] = (
        batch_path + "/" + prefix + df_caption.chunk_id + ".zip"
    )
    # TODO: i think unzipping the files inside a folder is the best choice ? so we dont have to modify the dataloader
    # alternatively modify the dataloader and not unzip the

    # create multiresolution caption
    training_df = create_tag_based_training_dataframe(
        dataframe=df_caption,
        image_width_col_name="image_width",
        image_height_col_name="image_height",
        caption_col="caption",  # caption column name
        bucket_batch_size=8,
        repeat_batch=10,
        seed=seed,
        max_res_areas=MAXIMUM_RESOLUTION_AREA,  # modify this if you want long or wide image
        bucket_lower_bound_resolutions=BUCKET_LOWER_BOUND_RESOLUTION,  # modify this if you want long or wide image
        extreme_aspect_ratio_clip=2.0,  # modify this if you want long or wide image
    )

    # rebundle all thread
    prefetcher_thread.join()

    # delete the current batch after training loop is done to prevent out of storage
    delete_file_or_folder(
        os.path.join(create_abs_path(ramdisk_path), f"{batch_name}{batch_number}")
    )
    # delete aria temp file url
    delete_file_or_folder(
        os.path.join(create_abs_path(ramdisk_path), f"{batch_name}{batch_number}.txt")
    )
    print()


main()
