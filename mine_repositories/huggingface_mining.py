from huggingface_hub import HfApi, list_models, ModelCard
import csv
from loguru import logger
import pandas as pd
import numpy as np


keywords = set()
with open("keywords.txt", "r") as f:
    for line in f:
        keywords.add(line.strip().lower())


def get_top_n_models(n):
    api = HfApi()
    models = api.list_models(sort='downloads', direction=-1)

    num = 0
    for model_info in models:
        try:
            if num < n:

                num += 1
                with open("huggingface_model_list1.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([model_info.id, model_info.downloads])
                logger.info(f"mining {num}")
            else:
                break
        except:
            logger.info(f"model {model_info.id} has problem")
            continue



if __name__ == "__main__":

    logger.add("hf_logs.log")
    get_top_n_models(2000)

