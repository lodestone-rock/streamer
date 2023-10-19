from typing import Union, Callable, Tuple
import zipfile
from PIL import ImageFile, Image
import pandas as pd
import numpy as np
import pathlib
from transformers import CLIPTokenizer
import cv2
import time
import os

from multiprocessing.dummy import Pool
# from batch_downloader import process_image_in_zip # TODO: move this to utils.py so its tidy

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

ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_image(
    rescale_size: Union[list, tuple],
    image: Image = None,
    image_path: str = None,
    upper_bound: int = 10,
    debug: bool = False,
) -> Union[np.array, tuple]:
    r"""
    scale the image resolution to predetermined resolution and return
    it as numpy

    args:
        rescale_size (:obj:`list` or `tuple`):
            width and height target
        image (:obj:`PIL.Image` defaults to `None`):
            image to process if `none` then it will try to read from `image_path`
        image_path (:obj:`str` defaults to `None`):
            path to file
        upper_bound (:obj:`int`, *optional*, defaults to 10):
            major axis bound (not important, just set it as high as possible)
        debug (:obj:`bool`, *optional*, defaults to `False`):
            will return tuple (np.array, PIL.Image)

    return: np.array or (np.array, PIL.Image)
    """
    if image == None:
        image = Image.open(image_path)

    image = image.convert("RGB")

    # find the scaling factor for each axis
    x_scale = rescale_size[0] / image.size[0]
    y_scale = rescale_size[1] / image.size[1]
    scaling_factor = max(x_scale, y_scale)

    # rescale image with scaling factor
    new_scale = [
        round(image.size[0] * scaling_factor),
        round(image.size[1] * scaling_factor),
    ]
    sampling_algo = PIL.Image.LANCZOS
    image = image.resize(new_scale, resample=sampling_algo)

    # get smallest and largest res from image
    minor_axis_value = min(image.size)
    minor_axis = image.size.index(minor_axis_value)
    major_axis_value = max(image.size)
    major_axis = image.size.index(major_axis_value)

    # warning
    if max(image.size) < max(rescale_size):
        print(
            f"[WARN] image {image_path} is smaller than designated batch, zero pad will be added"
        )

    if minor_axis == 0:
        # left and right same crop top and bottom
        top = (image.size[1] - rescale_size[1]) // 2
        bottom = (image.size[1] + rescale_size[1]) // 2

        # remainder add
        bottom_remainder = top + bottom
        # left, top, right, bottom
        image = image.crop((0, top, image.size[0], bottom))
    else:
        # top and bottom same crop the left and right
        left = (image.size[0] - rescale_size[0]) // 2
        right = (image.size[0] + rescale_size[0]) // 2
        # left, top, right, bottom
        image = image.crop((left, 0, right, image.size[1]))

    # cheeky resize to catch missmatch
    image = image.resize(rescale_size, resample=sampling_algo)
    # for some reason np flip width and height
    np_image = np.array(image)
    # normalize
    np_image = np_image / 127.5 - 1
    # height width channel to channel height weight
    np_image = np.transpose(np_image, (2, 0, 1))
    # add batch axis
    # np_image = np.expand_dims(np_image, axis=0)

    if debug:
        return (np_image, image)
    else:
        return np_image


def process_image_in_zip(
    rescale_size: Union[list, tuple],
    zip_file_path: str,
    image_name: str,
    upper_bound: int = 10,
    debug: bool = False,
) -> Union[np.array, tuple]:
    r"""
    scale the image resolution to predetermined resolution and return
    it as numpy

    args:
        rescale_size (:obj:`list` or `tuple`):
            width and height target
        image (:obj:`PIL.Image` defaults to `None`):
            image to process if `none` then it will try to read from `image_path`
        zip_file_path (:obj:`str`):
            path to zip file
        image_name (:obj:`str`):
            image name in zip file
        upper_bound (:obj:`int`, *optional*, defaults to 10):
            major axis bound (not important, just set it as high as possible)
        debug (:obj:`bool`, *optional*, defaults to `False`):
            will return tuple (np.array, PIL.Image)

    return: np.array or (np.array, PIL.Image)
    """

    with zipfile.ZipFile(zip_file_path, "r") as archive:
        # start=time.time()
        with archive.open(image_name) as filename:
            # stop=time.time()
            # print(stop-start)
            
            with Image.open(filename) as image:

                # with TimingContextManager("converting image to RGB"):
                #     image = image.convert("RGB")
                image = image.convert("RGB")

                # find the scaling factor for each axis
                x_scale = rescale_size[0] / image.size[0]
                y_scale = rescale_size[1] / image.size[1]
                scaling_factor = max(x_scale, y_scale)

                # rescale image with scaling factor
                new_scale = [
                    round(image.size[0] * scaling_factor),
                    round(image.size[1] * scaling_factor),
                ]
                # if scaling_factor > 1.0:
                #     sampling_algo = Image.LANCZOS
                # else:
                #     sampling_algo = Image.NEAREST
                sampling_algo = Image.LANCZOS
                # with TimingContextManager("lanczos rescale1"):
                #     image = image.resize(new_scale, resample=sampling_algo)
                image = image.resize(new_scale, resample=sampling_algo)

                # get smallest and largest res from image
                minor_axis_value = min(image.size)
                minor_axis = image.size.index(minor_axis_value)
                major_axis_value = max(image.size)
                major_axis = image.size.index(major_axis_value)

                # warning
                if max(image.size) < max(rescale_size):
                    print(
                        f"[WARN] image {image_path} is smaller than designated batch, zero pad will be added"
                    )

                if minor_axis == 0:
                    # left and right same crop top and bottom
                    top = (image.size[1] - rescale_size[1]) // 2
                    bottom = (image.size[1] + rescale_size[1]) // 2

                    # remainder add
                    bottom_remainder = top + bottom
                    # left, top, right, bottom
                    image = image.crop((0, top, image.size[0], bottom))
                else:
                    # top and bottom same crop the left and right
                    left = (image.size[0] - rescale_size[0]) // 2
                    right = (image.size[0] + rescale_size[0]) // 2
                    # left, top, right, bottom
                    image = image.crop((left, 0, right, image.size[1]))

                # # cheeky resize to catch missmatch
                # with TimingContextManager("lanczos rescale2"):
                #     image = image.resize(rescale_size, resample=sampling_algo)
                image = image.resize(rescale_size, resample=sampling_algo)
                # for some reason np flip width and height
                np_image = np.array(image)
                # normalize
                np_image = np_image / 127.5 - 1
                # height width channel to channel height weight
                np_image = np.transpose(np_image, (2, 0, 1))
                # add batch axis
                # np_image = np.expand_dims(np_image, axis=0)

                if debug:
                    return (np_image, image)
                else:
                    return np_image


def numpy_to_pil_and_save(np_image, output_path):
    # Convert from channel-height-width back to height-width-channel
    np_image = np.transpose(np_image, (1, 2, 0))
    
    # Denormalize
    np_image = (np_image + 1) * 127.5
    
    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(np_image.astype('uint8'))

    # Save the PIL image to the specified output path
    pil_image.save(output_path)
    

def crop_resize_image(image: np.ndarray, size: tuple):
    # Note: Height and width are wrong here.  But they are consistently wrong.
    # The code works. Don't touch it until and unless prepared to do a FULL refactor.
    # initial_shape = image.shape



    width, height, _ = image.shape # 1801, 1200
    target_width, target_height = size # 640, 384
    assert (target_width > target_height and width > height) or \
           (target_width < target_height and width < height) or \
           (target_width == target_height), \
           f"An image got sent to an inappropriate bucket.  The bucket size is {size}. The image shape is {image.shape}."


    width_scale = width / target_width
    height_scale = height / target_height
    scaling_factor = max(width_scale, height_scale)

    # interpolation = cv2.INTER_LANCZOS4
    if scaling_factor < 1.0:
        interpolation = cv2.INTER_LANCZOS4
    else:
        interpolation = cv2.INTER_AREA

    # Calculate the aspect ratio of the target size
    target_aspect_ratio = target_width / target_height # 640 / 384 = 1.6667

    # Calculate the aspect ratio of the input image
    image_aspect_ratio = width / height # 1801 / 1200 = 1.500008

    if target_aspect_ratio > image_aspect_ratio:
        # Crop the image vertically to match the target aspect ratio (reduce height)
        new_height = int(width / target_aspect_ratio) # 1801 / 1.66667 = 1081 (correct)
        top = (height - new_height) // 2 
        bottom = top + new_height
        image = image[:, top:bottom, :]
    elif target_aspect_ratio < image_aspect_ratio:
        # Crop the image horizontally to match the target aspect ratio (reduce width)
        new_width = int(height * target_aspect_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        image = image[left:right, :, :]
    # intermediate_shape = image.shape
    # Center crop and resize the image using cv2.resize
    # Don't forget that cv2.resize uses a flipped size for absolutely no good reason!
    # with TimingContextManager("lanczos rescale"):
    image = cv2.resize(image, dsize=(target_height, target_width), interpolation=interpolation)
    # if not os.path.exists("./test_image3.png"):
    #     cv2.imwrite("./test_image3.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32) / 127.5 - 1.0
    # print(f"Image of {initial_shape} cropped to {intermediate_shape} then resized to {image.shape}")
    return image


def cv2_process_image_in_zip(
    rescale_size: Union[list, tuple],
    zip_file_path: str,
    image_name: str,
    debug: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, cv2.Mat]]:
    r"""
    Scale the image resolution to predetermined resolution and return
    it as a numpy array.

    Args:
        rescale_size (:obj:`list` or `tuple`):
            Width and height target.
        zip_file_path (:obj:`str`):
            Path to the zip file.
        image_name (:obj:`str`):
            Image name in the zip file.
        upper_bound (:obj:`int`, *optional*, defaults to 10):
            Major axis bound (not important, just set it as high as possible).
        debug (:obj:`bool`, *optional*, defaults to `False`):
            Will return a tuple (np.ndarray, cv2.Mat).

    Returns:
        np.ndarray or (np.ndarray, cv2.Mat)
    """

    with zipfile.ZipFile(zip_file_path, "r") as archive:
        with archive.open(image_name) as file_bytes:
            with TimingContextManager("read from zip and converting image to RGB"):
                np_image = np.frombuffer(file_bytes.read(), np.uint8)
                
                image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return crop_resize_image(image, [rescale_size[1], rescale_size[0]])


def cv2_process_image(
    rescale_size: Union[list, tuple],
    image_path: str,
) -> np.ndarray:
    r"""
    Scale the image resolution to predetermined resolution and return
    it as a numpy array.

    Args:
        rescale_size (:obj:`list` or `tuple`):
            Width and height target.
        image_path (:obj:`str`):
            Path to the zip file.
    Return:
        np.ndarray
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return crop_resize_image(image, [rescale_size[1], rescale_size[0]])


def tokenize_text(
    tokenizer: CLIPTokenizer,
    text_prompt: list,
    max_length: int,
    batch_slice: int = 1,
) -> dict:
    r"""
    wraps huggingface tokenizer function with some batching functionality
    convert long token for example (1,1002) to (1,10,102)
    start and end token are extracted and reappended for each batch

    args:
        tokenizer (:obj:`CLIPTokenizer`):
            tokenizer class
        text_prompt (:obj:`list`):
            batch text to be tokenized
        max_length (:obj:`int`):
            maximum token before clipping
        batch_slice (:obj:`int`, *optional*, defaults to 1):
            if greater than 1 it will slice the token into batch evenly
            (max_length-2) must be divisible by this value

    return:
        dict:
            {"attention_mask": np.array, "input_ids": np.array}
    """

    # check
    assert (
        max_length - 2
    ) % batch_slice == 0, "(max_length-2) must be divisible by batch_slice"

    text_input = tokenizer(
        text=text_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )

    max_length = tokenizer.model_max_length
    if batch_slice > 1:
        # ###[stack input ids]### #
        value = text_input["input_ids"]
        # strip start and end token
        # [start, token1, token2, ..., end] to
        # [token1, token2, ..., tokenN]
        content = value[:, 1:-1].reshape(-1, batch_slice, max_length - 2)
        # store start and end token and then reshape it to be concatenated
        start = np.full(
            shape=(content.shape[0], content.shape[1], 1), fill_value=[value[:, 0][0]]
        )
        stop = np.full(
            shape=(content.shape[0], content.shape[1], 1), fill_value=[value[:, -1][0]]
        )
        # concat start and end token
        # from shape (batch, 75*3+2)
        # to shape (batch, 3, 77)
        new_value = np.concatenate([start, content, stop], axis=-1)
        text_input["input_ids"] = new_value

        # ###[stack attention mask]### #
        mask = text_input["attention_mask"]
        # strip start and end mask
        # [start, mask1, mask2, ..., end] to
        # [mask1, mask2, ..., maskN]
        content = mask[:, 1:-1].reshape(-1, batch_slice, max_length - 2)
        # store start and end mask and then reshape it to be concatenated
        start = np.full(
            shape=(content.shape[0], content.shape[1], 1), fill_value=[mask[:, 0][0]]
        )
        # concat start and end mask
        # from shape (batch, 75*3+2)
        # to shape (batch, 3, 77)
        new_value = np.concatenate([start, start, content], axis=-1)
        text_input["attention_mask"] = new_value

    return text_input


def generate_batch(
    process_image_fn: Callable[[str, tuple], np.array],
    tokenize_text_fn: Callable[[str, str, int], dict],
    tokenizer: CLIPTokenizer,
    dataframe: pd.DataFrame,
    folder_path: str,
    image_name_col: str,
    caption_col: str,
    caption_token_length: int,
    width_col: str,
    height_col: str,
    batch_slice: int = 1,
) -> dict:
    """
    generate a single batch for training.
    use this function in a for loop while swapping the dataframe batch
    depends on process_image and tokenize_text function

    args:
        process_image_fn (:obj:`Callable`):
            process_image function
        process_image_fn (:obj:`Callable`):
            tokenize_text function
        tokenizer (:obj:`CLIPTokenizer`):
            tokenizer class
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        folder_path (:obj:`str`):
            path to image folder
        image_name_col (:obj:`str`):
            column name inside dataframe filled with image names
        caption_col (:obj:`str`):
            column name inside dataframe filled with text captions
        caption_token_length (:obj:`int`):
            maximum token before clipping
        width_col (:obj:`str`):
            column name inside dataframe filled with bucket width of an image
        height_col (:obj:`str`):
            column name inside dataframe filled with bucket height of an image
        batch_slice (:obj:`int`, *optional*, defaults to 1):
            if greater than 1 it will slice the token into batch evenly
            (caption_token_length-2) must be divisible by this value
    return:
        dict:
            {
                "attention_mask": np.array,
                "input_ids": np.array,
                "pixel_values": np.array
            }
    """
    # count batch size
    batch_size = len(dataframe)
    batch_image = []

    # ###[process image]### #
    # process batch sequentialy
    for x in range(batch_size):
        # get image name and size from datadrame
        image_name = dataframe.iloc[x][image_name_col]
        width_height = [dataframe.iloc[x][width_col], dataframe.iloc[x][height_col]]

        # grab iamge from path and then process it
        image_path = pathlib.Path(folder_path, image_name)
        image = process_image_fn(image_path=image_path, rescale_size=width_height)

        batch_image.append(image)
    # stack image into neat array
    batch_image = np.stack(batch_image)
    # as contiguous array
    batch_image = np.ascontiguousarray(batch_image)

    # ###[process token]### #
    batch_prompt = dataframe.loc[:, caption_col].tolist()
    tokenizer_dict = tokenize_text_fn(
        tokenizer=tokenizer,
        text_prompt=batch_prompt,
        max_length=caption_token_length,
        batch_slice=batch_slice,
    )
    output = {}
    output["pixel_values"] = batch_image
    output["input_ids"] = tokenizer_dict.input_ids
    output["attention_mask"] = tokenizer_dict.attention_mask

    return output


def generate_batch_from_zip_files_concurrent(
    process_image_fn: Callable[[str, tuple], np.array],
    tokenize_text_fn: Callable[[str, str, int], dict],
    tokenizer: CLIPTokenizer,
    dataframe: pd.DataFrame,
    zip_path_col: str,
    image_name_col: str,
    caption_col: str,
    caption_token_length: int,
    width_col: str,
    height_col: str,
    batch_slice: int = 1,
) -> dict:
    """
    generate a single batch for training.
    use this function in a for loop while swapping the dataframe batch
    depends on process_image and tokenize_text function

    args:
        process_image_fn (:obj:`Callable`):
            process_image function
        process_image_fn (:obj:`Callable`):
            tokenize_text function
        tokenizer (:obj:`CLIPTokenizer`):
            tokenizer class
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        zip_path_col (:obj:`str`):
            olumn name inside dataframe filled with zip path of associated image
        image_name_col (:obj:`str`):
            column name inside dataframe filled with image names
        caption_col (:obj:`str`):
            column name inside dataframe filled with text captions
        caption_token_length (:obj:`int`):
            maximum token before clipping
        width_col (:obj:`str`):
            column name inside dataframe filled with bucket width of an image
        height_col (:obj:`str`):
            column name inside dataframe filled with bucket height of an image
        batch_slice (:obj:`int`, *optional*, defaults to 1):
            if greater than 1 it will slice the token into batch evenly
            (caption_token_length-2) must be divisible by this value
    return:
        dict:
            {
                "attention_mask": np.array,
                "input_ids": np.array,
                "pixel_values": np.array
            }
    """
    # count batch size
    batch_size = len(dataframe)
    batch_image = []

    # ###[process image]### #
    # process batch concurently

    def _process_image_parallel(dataframe, x, image_name_col, zip_path_col, width_col, height_col, process_image_fn):
        image_name = dataframe.iloc[x][image_name_col]
        zip_file_name = dataframe.iloc[x][zip_path_col]
        width_height = [dataframe.iloc[x][width_col], dataframe.iloc[x][height_col]]

        # load image from zip then put it in image processor
        image = process_image_fn(
            zip_file_path=zip_file_name, image_name=image_name, rescale_size=width_height
        )

        return image


    with TimingContextManager(message="image processing"):
        with Pool(batch_size) as pool:
            batch_image = pool.starmap(
                _process_image_parallel,
                [(dataframe, x, image_name_col, zip_path_col, width_col, height_col, process_image_fn) for x in range(batch_size)]
            )

        # for x in range(batch_size):
        #     # get image name and size from datadrame
        #     image = process_image_parallel(
        #         dataframe, x, image_name_col, zip_path_col, width_col, height_col, process_image_fn
        #     )

        #     batch_image.append(image)
        # stack image into neat array
        batch_image = np.stack(batch_image)
        # as contiguous array
        batch_image = np.ascontiguousarray(batch_image)

    # ###[process token]### #
    # with TimingContextManager(message="tokenizer process"):
    batch_prompt = dataframe.loc[:, caption_col].tolist()
    tokenizer_dict = tokenize_text_fn(
        tokenizer=tokenizer,
        text_prompt=batch_prompt,
        max_length=caption_token_length,
        batch_slice=batch_slice,
    )
    output = {}
    output["pixel_values"] = batch_image
    output["input_ids"] = tokenizer_dict.input_ids
    output["attention_mask"] = tokenizer_dict.attention_mask

    return output


def generate_batch_from_zip_files(
    process_image_fn: Callable[[str, tuple], np.array],
    tokenize_text_fn: Callable[[str, str, int], dict],
    tokenizer: CLIPTokenizer,
    dataframe: pd.DataFrame,
    zip_path_col: str,
    image_name_col: str,
    caption_col: str,
    caption_token_length: int,
    width_col: str,
    height_col: str,
    batch_slice: int = 1,
) -> dict:
    """
    generate a single batch for training.
    use this function in a for loop while swapping the dataframe batch
    depends on process_image and tokenize_text function

    args:
        process_image_fn (:obj:`Callable`):
            process_image function
        process_image_fn (:obj:`Callable`):
            tokenize_text function
        tokenizer (:obj:`CLIPTokenizer`):
            tokenizer class
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        zip_path_col (:obj:`str`):
            olumn name inside dataframe filled with zip path of associated image
        image_name_col (:obj:`str`):
            column name inside dataframe filled with image names
        caption_col (:obj:`str`):
            column name inside dataframe filled with text captions
        caption_token_length (:obj:`int`):
            maximum token before clipping
        width_col (:obj:`str`):
            column name inside dataframe filled with bucket width of an image
        height_col (:obj:`str`):
            column name inside dataframe filled with bucket height of an image
        batch_slice (:obj:`int`, *optional*, defaults to 1):
            if greater than 1 it will slice the token into batch evenly
            (caption_token_length-2) must be divisible by this value
    return:
        dict:
            {
                "attention_mask": np.array,
                "input_ids": np.array,
                "pixel_values": np.array
            }
    """
    # count batch size
    batch_size = len(dataframe)
    batch_image = []

    # ###[process image]### #
    # process batch sequentialy
    for x in range(batch_size):
        # get image name and size from datadrame
        image_name = dataframe.iloc[x][image_name_col]
        zip_file_name = dataframe.iloc[x][zip_path_col]
        width_height = [dataframe.iloc[x][width_col], dataframe.iloc[x][height_col]]

        # load image from zip then put it in image processor

        image = process_image_fn(
            zip_file_path=zip_file_name, image_name=image_name, rescale_size=width_height
            )

        batch_image.append(image)
    # stack image into neat array
    batch_image = np.stack(batch_image)
    # as contiguous array
    batch_image = np.ascontiguousarray(batch_image)

    # ###[process token]### #
    batch_prompt = dataframe.loc[:, caption_col].tolist()
    tokenizer_dict = tokenize_text_fn(
        tokenizer=tokenizer,
        text_prompt=batch_prompt,
        max_length=caption_token_length,
        batch_slice=batch_slice,
    )
    output = {}
    output["pixel_values"] = batch_image
    output["input_ids"] = tokenizer_dict.input_ids
    output["attention_mask"] = tokenizer_dict.attention_mask

    return output

def generate_batch_concurrent(
    process_image_fn: Callable[[str, tuple], np.array],
    tokenize_text_fn: Callable[[str, str, int], dict],
    tokenizer: CLIPTokenizer,
    dataframe: pd.DataFrame,
    image_name_col: str,
    caption_col: str,
    caption_token_length: int,
    width_col: str,
    height_col: str,
    batch_slice: int = 1,
) -> dict:
    """
    generate a single batch for training.
    use this function in a for loop while swapping the dataframe batch
    depends on process_image and tokenize_text function

    args:
        process_image_fn (:obj:`Callable`):
            process_image function
        process_image_fn (:obj:`Callable`):
            tokenize_text function
        tokenizer (:obj:`CLIPTokenizer`):
            tokenizer class
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        image_name_col (:obj:`str`):
            column name inside dataframe filled with ABSOLUTE image path
        caption_col (:obj:`str`):
            column name inside dataframe filled with text captions
        caption_token_length (:obj:`int`):
            maximum token before clipping
        width_col (:obj:`str`):
            column name inside dataframe filled with bucket width of an image
        height_col (:obj:`str`):
            column name inside dataframe filled with bucket height of an image
        batch_slice (:obj:`int`, *optional*, defaults to 1):
            if greater than 1 it will slice the token into batch evenly
            (caption_token_length-2) must be divisible by this value
    return:
        dict:
            {
                "attention_mask": np.array,
                "input_ids": np.array,
                "pixel_values": np.array
            }
    """
    # count batch size
    batch_size = len(dataframe)
    batch_image = []

    # ###[process image]### #
    # process batch concurently

    def _process_image_parallel(dataframe, x, image_name_col, width_col, height_col, process_image_fn):

        image_path = dataframe.iloc[x][image_name_col]
        width_height = [dataframe.iloc[x][width_col], dataframe.iloc[x][height_col]]

        # load image from zip then put it in image processor
        image = process_image_fn(image_path=image_path, rescale_size=width_height)
        return image

    # concurrent image proces since cv2 and numpy process are not GIL bound 
    with Pool(batch_size) as pool:
        batch_image = pool.starmap(
            _process_image_parallel,
            [(dataframe, x, image_name_col, width_col, height_col, process_image_fn) for x in range(batch_size)]
        )

        # stack image into neat array
        batch_image = np.stack(batch_image)
        # as contiguous array
        batch_image = np.ascontiguousarray(batch_image)

    # ###[process token]### #
    batch_prompt = dataframe.loc[:, caption_col].tolist()
    tokenizer_dict = tokenize_text_fn(
        tokenizer=tokenizer,
        text_prompt=batch_prompt,
        max_length=caption_token_length,
        batch_slice=batch_slice,
    )
    output = {}
    output["pixel_values"] = batch_image
    output["input_ids"] = tokenizer_dict.input_ids
    output["attention_mask"] = tokenizer_dict.attention_mask

    return output