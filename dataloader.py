# python
import os
from threading import Thread
import pandas as pd
from typing import Optional, List
from multiprocessing import Pool
import time

from batch_downloader import (
    read_json_file,
    # download_chunks_of_dataset,
    regex_search_list,
    concatenate_csv_files,
    create_abs_path,
    list_files_in_directory,
    delete_file_or_folder,
    prefetch_data,
    prefetch_data_with_validation,
    # create_batches_from_list,
    # check_image_error_in_zip,
    # flatten_list,
    # validate_files_in_parallel,
    # validate_downloaded_batch,
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
        creds_data:str,  # Replace Any with the actual type of creds_data
        ramdisk_path: str,
        batch_number: int,  # This should be incremented for each successful data loading
        seed: int = 432,  # This should be incremented when all batches are processed
        batch_size: int = 2,
        numb_of_prefetched_batch: int = 2,
        maximum_resolution_areas: List[int] = [576**2, 704**2, 832**2, 960**2, 1088**2],
        bucket_lower_bound_resolutions: List[int] = [384, 512, 576, 704, 832],
        _batch_name: str = "batch_",
        _prefix: str = "16384-e6-",
    ):
        
        self.creds_data = creds_data
        self.ramdisk_path = ramdisk_path
        self.batch_number = batch_number
        self.seed = seed
        self.batch_size = batch_size
        self.numb_of_prefetched_batch = numb_of_prefetched_batch
        self.maximum_resolution_areas = maximum_resolution_areas
        self.bucket_lower_bound_resolutions = bucket_lower_bound_resolutions
        self._batch_name = _batch_name
        self._prefix = _prefix

        self.repo_id = read_json_file(create_abs_path(creds_data))



def main():
    creds_data = "repo.json"
    ramdisk_path = "ramdisk"
    repo_path = "chunks"
    batch_name = "batch_"
    batch_number = 0  # <<< this should be incremented for each sucessfull dataloading
    numb_of_prefetched_batch = 1
    seed = 42  # <<< this should be incremented when all batch is processed
    prefix = "16384-e6-"
    MAXIMUM_RESOLUTION_AREA = [576**2, 704**2, 832**2, 960**2, 1088**2]
    BUCKET_LOWER_BOUND_RESOLUTION = [384, 512, 576, 704, 832]

    # grab token and repo id from json file
    repo_id = read_json_file(create_abs_path(creds_data))

    # TODO: convert this to a class NO NEED :kek: 
    # when it initialized 

    # TODO:this download should run in a separate thread and have batch offset
    # no! actually run this function twice, first sequential then concurrent,
    # it wont download the current batch if the data already there. so it will "skips" to the dataloader part
    # after dataloader part is finished and the training loop finished just delete the current batch and increment
    # when it's incremented the download_chunks_of_dataset will try to download next batch but it's already there
    # so it skips again and the thread prefetch the next batch
    # download batch

    # TODO: add image check functionality for prefetched batch
    # not sure for the first batch tho i think i need to add counter to indicate if the first batch need to be checked
    # just add indicator file named "is_audited.txt" or something i think   

    # the prefetcher thread
    # the prefetcher ensures the next batch will be ready before the training loop even start

    # download multiple prefetched batch in advance 
    # get all required repo by looping through json
    # TODO: URGENT! test if batch increment is working as intended and has no overlaping file (which i doubt)
    for x in range(0,3):
        prefetch_data_with_validation(
            ramdisk_path=ramdisk_path,
            repo_name=repo_id[f"repo_{x}"]["name"],
            token=repo_id["token"],
            repo_path=repo_path,
            batch_number=batch_number,
            batch_size=repo_id[f"repo_{x}"]["file_per_batch"],
            numb_of_prefetched_batch=numb_of_prefetched_batch,
            seed=seed,
            prefix=repo_id[f"repo_{x}"]["prefix"],
            csv_filenames_col="filename",
            numb_of_validator_threads= 80 * 16,
            _debug_mode_validation=False,
        )

    # dataloader part also validation part
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
    



    # def generate_batch_wrapper(
    #     list_of_batch: list,
    #     queue: Queue,
    #     print_debug: bool = False):
    # # loop until queue is full
    #     for batch in list_of_batch:
    #         current_batch = generate_batch(
    #             process_image_fn=process_image,
    #             tokenize_text_fn=tokenize_text,
    #             tokenizer=tokenizer,
    #             dataframe=data_processed.iloc[batch *
    #                                             batch_size:batch*batch_size+batch_size],
    #             folder_path=image_folder,
    #             image_name_col=image_name_col,
    #             caption_col=caption_col,
    #             caption_token_length=token_length,
    #             width_col=width_height[0],
    #             height_col=width_height[1],
    #             tokenizer_path=model_dir,
    #             batch_slice=token_concatenate_count
    #         )
    #         if print_debug and queue.full():
    #             print("queue is full!")
    #         # put task in queue
    #         queue.put(current_batch)
    #         if print_debug:
    #             print(f"putting task {batch} into queue")




    # dataloader finish part

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
