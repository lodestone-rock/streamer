from transformers import CLIPTokenizer
from dataloader import DataLoader
import time
import gc

from utils import (
    numpy_to_pil_and_save,
    write_list_to_file,
    TimingContextManager,
)

# debug test stuff
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

dataloader = DataLoader(
    tokenizer_obj=tokenizer,
    config="repo2.json",  # Replace Any with the actual type of creds_data
    ramdisk_path="ramdisk",
    chunk_number=0,  # This should be incremented for each successful data loading
    seed=42,  # This should be incremented when all batches are processed
    training_batch_size=8,
    repeat_batch=5,
    maximum_resolution_areas=[
        576**2,
        704**2,
        832**2,
        960**2,
        1088**2,
    ],
    bucket_lower_bound_resolutions=[384, 512, 576, 704, 832],
    numb_of_worker_thread=20,
    queue_get_timeout=10,
)

dataloader._print_debug = False
dataloader.grab_and_prefetch_chunk(
    1
)  # TODO: chunk number should be defined here so the thread is not terminated i think?
dataloader.prepare_training_dataframe()
dataloader.create_training_dataframe()
dataloader._bulk_batch_count = 100  # debug limit to 100 batch
dataloader.dispatch_worker()
with TimingContextManager("total queue"):
    for count in range(int(dataloader.total_batch)):
        with TimingContextManager(f"queue latency at batch {count}"):
            test = dataloader.grab_next_batch()
            if test == "end_of_batch":
                break
            # try:
            text = []
            for x, token in enumerate(test["input_ids"]):
                text.append(
                    str(x)
                    + " === "
                    + tokenizer.decode(token.reshape(-1))
                    .replace("<|endoftext|>", "")
                    .replace("<|startoftext|>", "")
                )
            write_list_to_file(text, f"{count}.txt")

            for x, np_image in enumerate(test["pixel_values"]):
                numpy_to_pil_and_save(np_image, f"{count}-{x}-pil.png")

            # print(count, "shape", test["pixel_values"].shape)
            # # print("shape", test["input_ids"].shape)
            # except:
            #     print(f"batch {count} is none")
            if count % int(dataloader.total_batch):
                gc.collect()
            time.sleep(0.01)
print()
