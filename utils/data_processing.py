import pandas as pd
import csv
import json


def dialogue2note_csvtojsonl(csv_path, jsonl_path):
    """
    Converts a CSV file to a JSONL file format.

    Args:
    - csv_path (str): the path of the input CSV file.
    - jsonl_path (str): the path of the output JSONL file.

    Returns:
    - None.
    """
    with open(csv_path, "r") as csv_file, open(jsonl_path, "w") as jsonl_file:
        # create a CSV reader object
        csv_reader = csv.DictReader(csv_file)

        # loop through the rows in the CSV file
        for row in csv_reader:
            # convert the row to a JSON string

            output_json = {}
            output_json[
                "instruction"
            ] = "Given this dialouge between a doctor and a patient, generate a note that summarizes the conversation between them: "

            # delet the \n in the dialouge or just let the modle learn that as well?
            dialogue = row["dialogue"]
            dialogue = dialogue.replace("\n", "")
            output_json["input"] = dialogue
            output_json["output"] = row["section_text"]

            json_string = json.dumps(output_json)

            # write the JSON string to the output file
            jsonl_file.write(json_string + "\n")


if __name__ == "__main__":
    import os

    dialogue2note_csvtojsonl(
        f"{os.environ['PROJECT_ROOT']}/clef-medqa/dataset/TaskB/TaskB-TrainingSet.csv",
        f"{os.environ['PROJECT_ROOT']}/train_llama_lora.jsonl",
    )

    dialogue2note_csvtojsonl(
        f"{os.environ['PROJECT_ROOT']}/clef-medqa/dataset/TaskB/TaskB-ValidationSet.csv",
        f"{os.environ['PROJECT_ROOT']}/dev_llama_lora.jsonl",
    )
