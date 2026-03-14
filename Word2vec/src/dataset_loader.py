from datasets import load_dataset

def load_and_storage(path : str):
    with open(path, "w") as f:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for i in dataset:
            f.write(i["text"])

if __name__ == "__main__":
    load_and_storage("../Data/dataset")