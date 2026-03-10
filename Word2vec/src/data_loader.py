from datasets import load_dataset
import re

def load_and_storage():
    with open("../Data/dataset", "w") as f:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for i in dataset:
            f.write(i["text"])

if __name__ == "__mian__":
    load_and_storage()