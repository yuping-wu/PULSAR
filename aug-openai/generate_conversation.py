import time
from multiprocessing import Pool

import openai
import pandas as pd
from tqdm import tqdm

NOISE_HEADERS = ["BP: 130/68.", "BP: 130/68", "BP: 120/80", "SPO2: 98%.", "HR: 98.", "INSTRUCTIONS", "VITALS", "RR:18.",
                 "BP: 133/70.", "BP: 169/74."]

K = 1000
NUM_PROCESSES = 10

openai.api_type = "azure"
openai.api_base = "https://docs-test-001.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = ""
deployment_name = "gpt-35-turbo"

prompt = open("prompt.txt").read()
sample_convo_1 = open("example_conversation_1.txt").read()
sample_note_1 = open("example_note_1.txt").read()

train_df = pd.read_csv("TaskC-TrainingSet.csv")
val_df = pd.read_csv("TaskC-ValidationSet.csv")
train_df_notes = train_df["note"].values.tolist()

prompt_messages_history = [
    {"role": "system",
     "content": "You are a system capable of generating a doctor-patient conversation based on its corresponding medical note.\n%s" % prompt},
    # {"role": "user", "content": prompt},
    # {"role": "assistant",
    #  "content": "I will follow the instructions while generating the conversation based on the input note provided. Please provide the input medical note."},
    {"role": "user", "content": "[INPUT MEDICAL NOTE]" + sample_note_1},
    {"role": "assistant", "content": "[OUTPUT CONVERSATION]" + sample_convo_1},
]


def calculate_score_match(note, headers):
    """
    Method to calculate number of headers matched in the medical note
    :param note: str
    :param headers: list[str]
    :return: int
    """
    score = 0
    for header in headers:
        if header in note:
            score += 1

    return score


def get_generated_conversation(note, prompt_messages):
    """
    Method to generate the conversation for a given medical note
    :param note: str
    :param prompt_messages: list[dict]
    :return: (str, boolean)
    """
    final_prompt_messages = prompt_messages + \
                            [{"role": "user",
                              "content": "[INPUT MEDICAL NOTE]%s" % note}]
    try:
        response = openai.ChatCompletion.create(
            temperature=0.75,
            model=deployment_name,
            messages=final_prompt_messages
        )
        generated_conversation = response["choices"][0]["message"]["content"]

        if "[doctor]" not in generated_conversation or "[patient]" not in generated_conversation:
            return note, ""

        generated_conversation = clean_artifacts_from_generated_conversation(generated_conversation)
        return note, generated_conversation

    except:
        return note, ""


def clean_artifacts_from_generated_conversation(convo):
    """
    Clean the generated conversation by only retaining content from the first [doctor] tag
    :param convo: str
    :return: str
    """
    start_index = convo.lower().index("[doctor]")
    cleaned_response = convo[start_index:].lower()
    return cleaned_response


def clean_mt_note(med_note):
    """
    Clean formatting issues in the mt-notes dataset
    :param med_note: str
    :return: str
    """
    med_note = med_note.replace(".,", ".\n").replace(" ,", " \n").replace(":,", ":\n")
    med_note = "\n".join([x.strip() for x in med_note.split("\n")])
    return med_note


if __name__ == "__main__":

    # gather headers present in the training set notes using a simple heuristic
    # Section headers are usually single to multi tokens with all upper caps and no other content present
    # Ex: CHIEF COMPLAINT:\n Patient had neck pain
    # Iterate across each line and check if the entire line is upper case.
    header_collections = []

    for note in tqdm(train_df_notes):
        note_lines = note.split("\n")
        for item in note_lines:
            if item.isupper():
                if item.strip() not in header_collections:
                    header_collections.append(item.strip())

    header_collections = [x for x in header_collections if x not in NOISE_HEADERS]

    # Read the csv with free medical notes available online from MTSamples made available on Kaggle
    mt_samples = pd.read_csv("mtsamples.csv")
    mt_samples = mt_samples.dropna()
    mt_samples = mt_samples.drop_duplicates()
    mt_samples_transcriptions = mt_samples["transcription"].values.tolist()
    mt_samples_ids = list(range(0, len(mt_samples_transcriptions)))

    # Score the medical notes in terms of the number of sections present in it based on the headers
    # extracted from the training set. Pick the top K medical notes based on the score.

    mt_note_score = {}
    for item in mt_samples_transcriptions:
        score = calculate_score_match(item, header_collections)
        if score > 0:
            mt_note_score[item] = score

    sorted_dict = dict(sorted(mt_note_score.items(), key=lambda item: item[1], reverse=True))

    # Extract the top K notes with highest scores
    top_k_notes = list(mt_note_score.keys())[:K]

    # EHRs from MTSamples downloaded through kaggle have an issue with the formatting.
    # New lines seem to be replaced with commas. Heuristically fixing this issue to get
    # a formatting that is similar to the training documents. We replace " ,", ":," cases with

    top_k_notes_formatted = [clean_mt_note(x) for x in top_k_notes]

    prompt_dialogue = []
    prompt_notes = []
    arguments = list(zip(top_k_notes_formatted[0:100], [prompt_messages_history] * 100))

    # for note in tqdm(top_k_notes_formatted[0:1]):
    #     _, generated_dialogue = get_generated_conversation(note, prompt_messages)
    #     generated_dialogue = clean_artifacts_from_generated_conversation(generated_dialogue)
    #     prompt_dialogue.append(generated_dialogue)
    #     prompt_notes.append(note)

    start_time = time.time()
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.starmap(get_generated_conversation, arguments)
        results = [x for x in results if x[-1] != ""]
        dataframe_data = pd.DataFrame(results)
        dataframe_data.columns = ["Notes", "Dialogue"]
        dataframe_data.to_csv("sample_generated_data.csv", index=False)
    end_time = time.time()

    print(end_time - start_time)
