import time

from datasets import load_dataset


def concat_a(data: dict):
    all_sentences = []
    for split in data.keys():
        for i in range(1):
            for lang_key in data[split]["translation"][i].keys():
                all_sentences.append(data[split]["translation"][i][lang_key])
    return all_sentences


def concat_b(data: dict):
    all_sentences = []
    for split in data.keys():
        for i in range(1):
            all_sentences.extend(data[split]["translation"][i].values())
    return all_sentences


def concat_c(data: dict):
    all_sentences = []

    return all_sentences


def main() -> None:
    """
    Main function.
    """
    data = load_dataset("iwslt2017", "iwslt2017-de-en")
    print(f"Data: {data.values()}\n")

    start_time = time.perf_counter()
    all_sentences = concat_b(data)
    end_time = time.perf_counter()
    print(
        f"Elapsed time: {end_time - start_time:.2f} seconds\nSentences: "
        f"{all_sentences}\n"
    )


if __name__ == "__main__":
    main()
