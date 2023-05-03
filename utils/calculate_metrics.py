import evaluate
import pandas as pd
import fire


def calculate_metrics_taskb(preds_file: str, target_csv: str):
    """Calculate the metrics from files.
    preds_file: str
        File containing one prediction per line
    target_csv: str
        target_csv from the competition.
    """
    metric = evaluate.load("rouge")
    predictions = []
    references = []
    with open(preds_file) as preds_fp:
        for line in preds_fp:
            predictions.append(line)

    df = pd.read_csv(target_csv)
    df["source_text"] = df["dialogue"]
    df["target_text"] = df["section_header"] + "\n" + df["section_text"]
    df = df.applymap(str)

    references = df["target_text"].tolist()

    ## Convert both the target and references into smaller case
    predictions = list(map(str.lower, predictions))
    references = list(map(str.lower, references))

    metric = metric.compute(predictions=predictions, references=references)
    print(metric)
    return metric


if __name__ == "__main__":
    fire.Fire(calculate_metrics_taskb)
