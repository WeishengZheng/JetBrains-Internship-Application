from datasets import load_dataset

def load_and_storage():
    with open("../Data/dataset", "w") as f:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for i in dataset:
            f.write(i["text"])

if __name__ == "__main__":
    load_and_storage()