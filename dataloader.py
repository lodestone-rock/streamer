# python
import os
from threading import Thread
from queue import Queue, Empty
import pandas as pd
from typing import Optional, List
from multiprocessing.dummy import Pool
import time
import concurrent.futures


from utils import (
    read_json_file,
    save_dict_to_json,
    regex_search_list,
    flatten_list,
    list_files_in_directory,
    create_batches_from_list,
    delete_file_or_folder,
    split_list,
)


from batch_downloader import (
    write_urls_to_file,
    concatenate_csv_files,
    prefetch_data_with_validation,
)

from dataframe_processor import (
    create_amplified_training_dataframe,
    shuffle,
)

from batch_processor import (
    process_image,
    tokenize_text,
    cv2_process_image,
    generate_batch,
)


class DataLoader:
    """
    Args:
        tokenizer_obj: Tokenizer object for caption tokenization.
        config (str): Path to a JSON configuration file containing dataset information.
        ramdisk_path (str): Path to store the dataset chunk, typically in a RAM disk.
        chunk_number (int): The current chunk number for data loading.
        seed (int, optional): Random number generator seed. Default is 432.
        training_batch_size (int, optional): Size of the training batch to load to the device. Default is 2.
        maximum_resolution_areas (List[int], optional): List of maximum image resolution areas. Default includes standard resolutions.
        bucket_lower_bound_resolutions (List[int], optional): List of minimum axis pixel counts for images.
        repeat_batch (int, optional): Number of times to repeat a batch during randomization. Default is 10.
        extreme_aspect_ratio_clip (float, optional): Maximum aspect ratio allowed for images. Default is 2.0.
        max_queue_size (int, optional): Maximum size of the data loading queue. Default is 100.
        context_concatenation_multiplier (int, optional): Multiplier for extending the context length. Default is 3.
        numb_of_worker_thread (int, optional): Number of worker threads for data loading. Default is 10.
        queue_get_timeout (int, optional): Timeout for waiting in the data loading queue. Default is 60.
    """

    def __init__(
        self,
        tokenizer_obj,
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
        repeat_batch: int = 10,
        extreme_aspect_ratio_clip: float = 2.5,
        max_queue_size=100,
        context_concatenation_multiplier: int = 3,
        numb_of_worker_thread: int = 10,
        queue_get_timeout: int = 60,
    ):
        # pass tokenizer object to tokenize captions
        self.tokenizer = tokenizer_obj
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
        #             "filename_col": "filename",
        #             "coma_separated_shuffle": True
        #         },
        #         "repo_1": {...},
        #         "repo_2": {...},
        #         ...
        #     },
        #     "token": "hf_token"
        # }
        self.config = read_json_file(config)

        # this defines how long it need to wait before declaring it's the end of batch
        # just in case dataloader is stalling
        self.queue_get_timeout = queue_get_timeout

        # training dataframe to be overriden either externaly or using a method
        # this will eventually be overriden by create_training_dataframe method
        self.training_dataframe = None

        # constant
        self.total_batch = 0
        self._first_batch_count = 0
        self._bulk_batch_count = 0
        self._width_col = "image_width"
        self._height_col = "image_height"
        self._caption_col = "caption"
        self._filename_col = "filename"
        self._filepath_col = "filepath"
        self._print_debug = True

        # CLIP text encoder special treatment to extend context length
        # how big you want the context length to be concatenated
        self.context_concatenation_multiplier = context_concatenation_multiplier
        # strip BOS and EOS token and then concatenate the content then cap off the begining and end with BOS and EOS
        # ie: 75 token * 3 concatenation + 2 (bos & eos)
        self.extended_context_length = (
            self.tokenizer.model_max_length - 2
        ) * self.context_concatenation_multiplier + 2

        # dataloader orchestration variable
        # define buffer to be stored to mask dadaloading latency
        self._queue = Queue(maxsize=max_queue_size)
        # repeat_batch is also defining factor on how much the worker thread is
        # not overridable for now / do not override it !
        self.numb_of_worker_thread = numb_of_worker_thread

    def grab_and_prefetch_chunk(self, numb_of_prefetched_batch: int = 1, chunk_number: int = None) -> None:
        """
        this will try to grab a chunk of dataset while also prefetch extra chunk for the next round
        this will download batch of zip files and in one chunk can have multiple zip files and csv

        args:
        numb_of_prefetched_batch (`int`) (default:`1`): how many prefetched chunk is downloaded conccurently.
        chunk_number (`int`) (default:`None`): grab this chunk number (make sure it's valid chunk! there's no check if it's invalid).
        """
        if chunk_number: #override if exist
            self.chunk_number = chunk_number
        repo_details = self.config["repo"]
        for repo in repo_details.keys():
            prefetch_data_with_validation(
                ramdisk_path=self.ramdisk_path,
                repo_name=repo_details[repo]["name"],
                token=self.config["token"],
                repo_path=repo_details[repo]["folder_path_in_repo"],
                batch_number=self.chunk_number,  # chunk at
                batch_size=repo_details[repo][
                    "file_per_batch"
                ],  # numb of zip and csv to download, should've renamed it
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
                self.ramdisk_path,
                f"{repo_details[repo]['prefix']}{self.chunk_number}",
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
            # renaming column to ensure consitency
            df_caption = df_caption.rename(
                columns={
                    repo_details[repo]["image_width_col_name"]: self._width_col,
                    repo_details[repo]["image_height_col_name"]: self._height_col,
                    repo_details[repo]["caption_col"]: self._caption_col,
                    repo_details[repo]["filename_col"]: self._filename_col,
                }
            )
            df_caption = df_caption.loc[
                :,
                [
                    self._width_col,
                    self._height_col,
                    self._caption_col,
                    self._filename_col,
                ],
            ]
            # create zip file path for each image to indicate where the image resides inside the zip
            # this 'image' dir is the default dir folder that created by prefetch_data_with_validation function
            df_caption[self._filepath_col] = (
                chunk_path + "/" + "image" + "/" + df_caption[self._filename_col]
            )

            # suffle caption if coma_separated_shuffle flag is true
            # ie: this, is, tag, based, caption
            if repo_details[repo]["coma_separated_shuffle"]:
                df_caption[self._caption_col] = df_caption[self._caption_col].apply(
                    lambda x: shuffle(x, self.seed)
                )

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
        result = create_amplified_training_dataframe(
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
        self.total_batch = result[1] + result[2]

    def _generate_batch_wrapper(self, batch_number: int, print_debug: bool = False):
        """
        this hidden method is being used by dispatch_worker method to generate dataset
        this hidden method push batch of numpy image and token to class internal queue (_queue)
        """
        # batch_numbers is just a number, it indicates the batch to retrieve from
        # loop until queue is full
        # slice the dataframe to only grab specific resolution bucket

        # this is O(1) so there has to be something that gradually accumulating
        dataset_slice = self.training_dataframe.iloc[
            batch_number
            * self.training_batch_size : batch_number
            * self.training_batch_size
            + self.training_batch_size
        ]
        try:
            start = time.time()
            current_batch = generate_batch(
                process_image_fn=cv2_process_image,  # function to process image
                tokenize_text_fn=tokenize_text,  # function to do tokenizer wizardry
                tokenizer=self.tokenizer,  # tokenizer object
                dataframe=dataset_slice,  # a batch slice
                image_name_col=self._filepath_col,  # a column where image path is stored
                caption_col=self._caption_col,
                caption_token_length=self.extended_context_length,
                width_col="new_image_width",  # this is default col name generated by create_tag_based_training_dataframe
                height_col="new_image_height",  # this is default col name generated by create_tag_based_training_dataframe
                batch_slice=self.context_concatenation_multiplier,
            )
            return current_batch

            stop = time.time()
            if print_debug:
                print(
                    f"creation of batch {batch_number} took {round(stop-start,4)} seconds"
                )

        except Exception as e:
            # self._queue.put(None) # TODO: skip queue if none
            print(f"skipping batch {batch_number} because of this error: {e}")
            return None

    def dispatch_worker(self):
        """
        this method will spawn multiple threads to create batch and put the result in internal queue (_queue)
        use grab_next_batch method to get the value from it
        """

        # grab first batch to be processed concurently
        first_batch = list(range(int(self._first_batch_count)))
        first_batch_args = [(x, self._print_debug) for x in first_batch]
        # first order dispatch
        # this will dispatch the first few resolution to ensure JAX compiled everything
        # just in case the compiled model is not using AOT compilation
        # forking thread
        with Pool(int(self._first_batch_count)) as pool:
            pool.starmap(func=self._generate_batch_wrapper, iterable=first_batch_args)
        # thread rebundled

        # convert the count of bulk batch into a list with offset of the first batch!
        bulk_batch = list(
            range(int(self._first_batch_count), int(self._bulk_batch_count))
        )
        bulk_batch_args = [(x, self._print_debug) for x in bulk_batch]
        # dice it into chunks of repeat batch, this ultimately will ensure chunking order for the majority of the batch
        # the tail batch is not guareanted however
        bulk_batch_args = split_list(bulk_batch_args, self.repeat_batch)

        # wrap logic into 1 function and run it under thread pool
        # so the code is not blocking here
        def _repeat_batch_thread_bundle(batch_of_batches: list):
            bundle = []
            for batches in batch_of_batches:
                bundle.append(self._generate_batch_wrapper(*batches))

            for batch in bundle:
                self._queue.put(batch)

        executor = concurrent.futures.ThreadPoolExecutor(self.numb_of_worker_thread)
        for batch_of_batches in bulk_batch_args:
            executor.submit(_repeat_batch_thread_bundle, batch_of_batches)

    def grab_next_batch(self):
        """
        this method will try to get the next batch from the internal queue
        it will return either batch value, None, or "end_of_batch" string
        "end_of_batch" will be returned if `queue_get_timeout` is reached to prevent stalling
        """
        try:
            batch = self._queue.get(timeout=self.queue_get_timeout)
        except Empty:
            print(
                f"there's no batch for {self.queue_get_timeout} assuming it's the end of the batch"
            )
            batch = "end_of_batch"

        return batch


    def delete_prev_chunks(self, prev_chunk: int):
        """
        use this method to delete previous chunk cache
        """
        try:
            repo_details = self.config["repo"]
            for repo in repo_details.keys():
                chunk_path = os.path.join(
                    self.ramdisk_path,
                    f"{repo_details[repo]['prefix']}{prev_chunk}",
                )
                aria_txt = chunk_path + ".txt"
                delete_file_or_folder(chunk_path)
                delete_file_or_folder(aria_txt)
        except Exception as e:
            print(f"deletion error: {e}")