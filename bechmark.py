import os
from multiprocessing.dummy import Pool
import cv2
import numpy as np
from .batch_downloader import list_files_in_zip
import time
import zipfile
from PIL import Image


def read_image_from_directory(image_path):
    with TimingContextManager("opening image directly"):
        try:
            # Read the image using OpenCV
            image = cv2.imread(image_path)

            # Check if the image was loaded successfully
            if image is not None:
                # Convert the image to a NumPy array
                image_np = np.asarray(image)
                return image_np
            else:
                print(f"Failed to read the image from {image_path}")
                return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


def read_image_from_zip(zip_file_path, image_filename):
    with TimingContextManager("opening image in zip"):
        try:
            # Open the zip file
            with zipfile.ZipFile(zip_file_path, "r") as zip_file:
                # Read the image from the zip file
                with zip_file.open(image_filename) as image_file:
                    # Read the image as binary data
                    image_data = image_file.read()

                    # Convert the binary data to a NumPy array
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # Check if the image was loaded successfully
                    if image is not None:
                        return image
                    else:
                        print(
                            f"Failed to read the image {image_filename} from the zip file"
                        )
                        return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


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


if __name__ == "__main__":
    # List of filenames to process
    directory_path = "ramdisk/16384-e6-0/zero"
    zip_path = "ramdisk/16384-e6-0/16384-e6-d9d3cd92-fc95-42ff-bfa8-80879461a410.zip"
    filenames = os.listdir(directory_path)[:1000]

    # Number of threads (adjust this according to your needs)
    num_threads = len(filenames)

    # Create a thread pool
    pool = Pool(200)

    # Use the thread pool to process the filenames
    with TimingContextManager("opening image directly overall took"):
        image = pool.map(
            read_image_from_directory,
            [os.path.join(directory_path, filename) for filename in filenames],
        )

        # Close the thread pool
        pool.close()
        pool.join()

    pool = Pool(200)

    with TimingContextManager("opening image in zip overall took"):
        image = pool.starmap(
            read_image_from_zip, [(zip_path, filename) for filename in filenames]
        )

        # Close the thread pool
        pool.close()
        pool.join()

    print()
