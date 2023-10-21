# python
import os
from threading import Thread
from queue import Queue
import pandas as pd
from typing import Optional, List
import gc
# from multiprocessing import  Process, Queue
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
    generate_batch_from_zip_files,
    process_image,
    tokenize_text,
    process_image_in_zip,
    cv2_process_image_in_zip,
    numpy_to_pil_and_save,
    generate_batch_from_zip_files_concurrent,
    cv2_process_image,
    generate_batch_concurrent
)

from transformers import CLIPTokenizer

class TimingContextManager:
    def __init__(self, message:str=""):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        execution_time = end_time - self.start_time
        print(f"{self.message} took {execution_time} seconds to execute.")


# TODO: should i put stuff in class or i can write a neat function instead
class DataLoader:
    def __init__(
        self,
        config: str,  # Replace Any with the actual type of creds_data
        ramdisk_path: str,
        chunk_number: int,  # This should be incremented for each successful data loading
        seed: int = 432,  # This should be incremented when all batches are processed
        training_batch_size: int = 2,
        maximum_resolution_areas: List[int] = [
            576**2,
            704**2,
            832**2,
            960**2,
            1088**2,
        ],
        bucket_lower_bound_resolutions: List[int] = [384, 512, 576, 704, 832],
        repeat_batch:int = 10,
        extreme_aspect_ratio_clip:float=2.0
    ):  
        # storage path to store the dataset chunk, preferably ramdisk hence the attribute name 
        self.ramdisk_path = ramdisk_path
        # curent chunk number, please keep the seed constant
        # it will try to grab a next chunk from dataset repo
        # if you increment the seed please reset this hunk number to 0 to start over
        # increment the chunk number to grab the next chunk
        self.chunk_number = chunk_number
        # rng seed 
        self.seed = seed
        # how big is the training batch loaded to device at a time
        self.training_batch_size = training_batch_size
        # absolute limit of the image size in area
        self.maximum_resolution_areas = maximum_resolution_areas
        # a set of minimum axis pixel count either height or width
        self.bucket_lower_bound_resolutions = bucket_lower_bound_resolutions
        # repeat batch during randomizing to ensure the next batch has the same res as previous one
        # this to prevent jax swapping the cache too often which cause performace degradation when
        # dealing with variable input array size
        self.repeat_batch = repeat_batch
        self.extreme_aspect_ratio_clip = extreme_aspect_ratio_clip
        # read configuration from json creds
        # the config file format
        # {
        #     "repo": {
        #         "repo_0": {
        #             "name": "planetexpress/e6",
        #             "prefix": "16384-e6-",
        #             "total_file_count": -1,
        #             "file_per_batch": 1
        #             "folder_path_in_repo": "chunks"
        #             "image_width_col_name": "image_width"
        #             "image_height_col_name": "image_height"
        #             "caption_col": "caption"
        #             "filename_col": "filename"
        #         },
        #         "repo_1": {...},
        #         "repo_2": {...},
        #         ...
        #     },
        #     "token": "hf_token"
        # }
        self.config = read_json_file(create_abs_path(config))
        # training dataframe to be overriden either externaly or using a method
        # this will eventually be overriden by create_training_dataframe method        
        self.training_dataframe = None

        # constant
        self._first_batch_count = 0
        self._bulk_batch_count = 0
        self._width_col = "image_width"
        self._height_col = "image_height"
        self._caption_col = "caption"
        self._filename_col = "filename"

    def grab_and_prefetch_chunk(self, numb_of_prefetched_batch:int=1) -> None:
        """
        this will try to grab a chunk of dataset while also prefetch extra chunk for the next round
        this will download batch of zip files and in one chunk can have multiple zip files and csv

        args:
        numb_of_prefetched_batch (`int`) (default:`1`): how many prefetched chunk is downloaded conccurently.
        """
        repo_details = self.config["repo"]
        for repo in repo_details.keys():
            prefetch_data_with_validation(
                ramdisk_path=self.ramdisk_path,
                repo_name=repo_details[repo]["name"],
                token=self.config["token"],
                repo_path=repo_details[repo]["folder_path_in_repo"],
                batch_number=self.chunk_number, # chunk at 
                batch_size=repo_details[repo]["file_per_batch"], # numb of zip and csv to download, should've renamed it
                numb_of_prefetched_batch=numb_of_prefetched_batch,
                seed=self.seed,
                prefix=repo_details[repo]["prefix"],
            )

    def prepare_training_dataframe(self) -> None:
        """
        this method is a custom method please overwrite it to your need
        or just assign `training_dataframe` attributes directly with your dataframe as long it meets this format:

        """
        dfs = []

        # i am assuming the csv is identical (which is not!)
        repo_details = self.config["repo"]
        for repo in repo_details.keys():
            chunk_path = os.path.join(
                create_abs_path(self.ramdisk_path), 
                f"{repo_details[repo]['prefix']}{self.chunk_number}"
            )
            file_list = list_files_in_directory(chunk_path)
            # get the csvs and convert it to abs path
            csvs = regex_search_list(file_list, r".csv")
            csvs = [os.path.join(chunk_path, csv) for csv in csvs]
            # get the zip and convert it to abs path
            zips = regex_search_list(file_list, r".zip")
            zips = [os.path.join(chunk_path, zip) for zip in zips]

            # combine csvs into 1 dataframe
            df_caption = concatenate_csv_files(csvs)
            # create zip file path for each image to indicate where the image resides inside the zip
            df_caption["filepaths"]=chunk_path+"/"+"image"+"/"+df_caption[repo_details[repo]["filename_col"]]
            df_caption = df_caption.rename(columns={
                repo_details[repo]["image_width_col_name"]: self._width_col,
                repo_details[repo]["image_height_col_name"]: self._height_col,
                repo_details[repo]["caption_col"]: self._caption_col,
                repo_details[repo]["filename_col"]: self._filename_col,
                })
            dfs.append(df_caption)
        dfs = pd.concat(dfs, axis=0)
        self.training_dataframe = dfs

    def create_training_dataframe(self) -> None:
        """
        this method can be overriden by other method suitable to construct the dataframe
        this method overrides the training_dataframe attributes by 
        creating virtual resolution bucket either by downscaling or upscaling.
        this method also ensures the first batch of the dataset is a unique batch 
        """

        # this returns tuple (dataframe, first_batch_count, bulk_batch_count)
        result = create_tag_based_training_dataframe(
            dataframe=self.training_dataframe,
            image_width_col_name=self._width_col,
            image_height_col_name=self._height_col,
            caption_col=self._caption_col,  # caption column name
            bucket_batch_size=self.training_batch_size,
            repeat_batch=self.repeat_batch,
            seed=self.seed,
            max_res_areas=self.maximum_resolution_areas,  # modify this if you want long or wide image
            bucket_lower_bound_resolutions=self.bucket_lower_bound_resolutions,  # modify this if you want long or wide image
            extreme_aspect_ratio_clip=self.extreme_aspect_ratio_clip,  # modify this if you want long or wide image
        )
        self.training_dataframe = result[0]
        self._first_batch_count = result[1]
        self._bulk_batch_count = result[2]
    
    def prepare_worker_threads(self):
        # TODO: put the thread here and create a method which is a simple iterator that periodically grab from the queue
        pass

def main():
    creds_data = "repo.json"
    repo_id = read_json_file(create_abs_path(creds_data))
    ramdisk_path = "ramdisk"
    repo_path = "chunks"
    batch_name = "16384-e6-"  # <<< TODO: change this im debbuging stuff
    batch_number = 0  # <<< this should be incremented for each sucessfull dataloading
    numb_of_prefetched_batch = 1
    bucket_batch_size = 8*10
    seed = 42  # <<< this should be incremented when all batch is processed
    prefix = "16384-e6-"  # <<< TODO: change this im debbuging stuff
    MAXIMUM_RESOLUTION_AREA = [576**2, 704**2, 832**2, 960**2, 1088**2]
    BUCKET_LOWER_BOUND_RESOLUTION = [384, 512, 576, 704, 832]

    # grab token and repo id from json file

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
    for x in range(0, 1):
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
            # csv_filenames_col="filename",
            # numb_of_validator_threads=80 * 16,
            # _disable_validation=True
            # _debug_mode_validation=False,
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
    df_caption["filepaths"]=batch_path+"/"+"image"+"/"+df_caption["filename"]


    # create multiresolution caption
    training_df = create_tag_based_training_dataframe(
        dataframe=df_caption,
        image_width_col_name="image_width",
        image_height_col_name="image_height",
        caption_col="caption",  # caption column name
        bucket_batch_size=bucket_batch_size,
        repeat_batch=10,
        seed=seed,
        max_res_areas=MAXIMUM_RESOLUTION_AREA,  # modify this if you want long or wide image
        bucket_lower_bound_resolutions=BUCKET_LOWER_BOUND_RESOLUTION,  # modify this if you want long or wide image
        extreme_aspect_ratio_clip=2.0,  # modify this if you want long or wide image
    )
    
    # debug to check unique value
    training_df["combined"]=training_df["new_image_width"].astype(str)+","+training_df["new_image_height"].astype(str)
    print(training_df.combined.value_counts())

    # TODO: run generate batch wrapper and simulate training
    # the put giant try catch block in generate_batch_wrapper so it just skips a batch that has broken image
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


    #### adopted from messy training script
    def generate_batch_wrapper(
        list_of_batch: list, queue: Queue, print_debug: bool = False
    ):
        # loop until queue is full
        for batch in list_of_batch:
            try:
                start = time.time()
                current_batch = generate_batch_concurrent(
                    process_image_fn=cv2_process_image,  # function to process image
                    tokenize_text_fn=tokenize_text,  # function to do tokenizer wizardry
                    tokenizer=tokenizer,  # tokenizer object
                    dataframe=training_df.iloc[
                        batch * bucket_batch_size : batch * bucket_batch_size + bucket_batch_size # slice the dataframe to only grab that resolution
                    ], # a batch slice 
                    image_name_col="filepaths",
                    caption_col="caption",
                    caption_token_length=75 * 3 + 2,
                    width_col="new_image_width",
                    height_col="new_image_height",
                    batch_slice=3,
                )
                if print_debug and queue.full():
                    print("queue is full!")
                # put task in queue
                queue.put(current_batch)
                stop = time.time()
                if print_debug:
                    print(f"putting {bucket_batch_size} images into {batch} queue took {round(stop-start,4)} seconds")
            
            except Exception as e:
                queue.put(None) # TODO: skip queue if none
                print(f"skipping batch {batch} because of this error: {e}")
        
        gc.collect()

    # get group index as batch order
    assert (
        len(training_df) % bucket_batch_size == 0
    ), f"DATA IS NOT CLEANLY DIVISIBLE BY {bucket_batch_size} {len(training_df)%bucket_batch_size}"
    batch_order = list(range(0, len(training_df) // bucket_batch_size))[:50]

    # store training array here
    batch_queue = Queue(maxsize=100)

    # spawn another process for processing images
    batch_processors = []
    proces_count = 10
    for t in range(proces_count):
        batch_processor = Thread(
            target=generate_batch_wrapper,
            args=[
                # a slice of dataframe batch_order
                batch_order[t*(len(batch_order)//proces_count):t*(len(batch_order)//proces_count)+(len(batch_order)//proces_count)], 
                batch_queue, 
                False]
        )
        batch_processors.append(batch_processor)


    # spawn process
    for x in batch_processors:
        x.start()
    # dataloader finish part

    # digest the batch
    with TimingContextManager("total queue"):
        for count, outer in enumerate(batch_order):
            with TimingContextManager("queue latency"):
                t = batch_queue.get() # if none skip!

                print(f"batch {count} of {len(batch_order)}")
                if count == len(batch_order):
                    break
                if t == None:
                    continue
                # time.sleep(0.01)
                # for x, np_image in enumerate(t["pixel_values"]):
                #     numpy_to_pil_and_save(np_image, f"{outer}-{x}-pil.png")

    # delete the current batch after training loop is done to prevent out of storage
    delete_file_or_folder(
        os.path.join(create_abs_path(ramdisk_path), f"{batch_name}{batch_number}")
    )
    # delete aria temp file url
    delete_file_or_folder(
        os.path.join(create_abs_path(ramdisk_path), f"{batch_name}{batch_number}.txt")
    )
    print()


# main()

dataloader = DataLoader(
    config="repo2.json",  # Replace Any with the actual type of creds_data
    ramdisk_path="ramdisk",
    chunk_number=0,  # This should be incremented for each successful data loading
    seed=42,  # This should be incremented when all batches are processed
    training_batch_size=8*1,
    maximum_resolution_areas=[
        576**2,
        704**2,
        832**2,
        960**2,
        1088**2,
    ],
    bucket_lower_bound_resolutions=[384, 512, 576, 704, 832],
)

dataloader.grab_and_prefetch_chunk(1)
dataloader.prepare_training_dataframe()
dataloader.create_training_dataframe()
print()