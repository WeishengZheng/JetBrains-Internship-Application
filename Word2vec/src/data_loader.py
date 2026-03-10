from datasets import load_dataset
import re

def load_and_storage():
    try:
        f = open("../Data/dataset", "w")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for i in dataset:
            f.write(i["text"])
    finally:
        f.close()

if __name__ == "__mian__":
    load_and_storage()