"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import sys
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tqdm
from rouge_score import rouge_scorer
import utils

import fire


topic_map = {
    "fam/sochx" : "FAMILY HISTORY/SOCIAL HISTORY",
    "genhx" : "HISTORY of PRESENT ILLNESS",
    "pastmedicalhx" : "PAST MEDICAL HISTORY",
    "cc" : "CHIEF COMPLAINT",
    "pastsurgical" : "PAST SURGICAL HISTORY",
    "allergy": None,
    "ros" : "REVIEW OF SYSTEMS",
    "medications": None,
    "assessment": None,
    "exam": None,
    "diagnosis": None,
    "disposition": None,
    "plan": None,
    "edcourse" : "EMERGENCY DEPARTMENT COURSE",
    "immunizations": None,
    "imaging": None,
    "gynhx" : "GYNECOLOGIC HISTORY",
    "procedures": None,
    "other_history": None,
    "labs": None,
}

def encode_prompt_d2s(prompt_instructions):
    """Encode prompt from data to section of note.
        section_header	section_text	dialogue
    Args:
        prompt_instructions: Examples basically

    Returns:
        str: Encoded prompt with examples
    """
    

    prompt = open("aug-openai/prompt-d2s.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions[:-1]):
        input, topic, output = task_dict["dialogue"], task_dict["section_header"], task_dict["section_text"]
        topic = topic_map[topic.lower()] or topic.replace('_', ' ')
        topic = topic.lower().capitalize()
        # instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        # input = "<noinput>" if input.lower() == "" else input
        prompt += f"Example Conversation about the patient's {topic}:\n{input}\n"
        prompt += f"Example Output Note:\n{output}\n*END*\n\n"

    input, topic, output = prompt_instructions[-1]["dialogue"], prompt_instructions[-1]["section_header"], prompt_instructions[-1]["section_text"]
    topic_str = topic_map[topic.lower()] or topic.replace('_', ' ')
    topic_str = topic.lower().capitalize()
    prompt += f"Input Conversation about the patient's {topic_str}:\n{input}\n"
    prompt += f"Output Note:"
    print(prompt)
    return prompt, (input, topic)

def encode_prompt_d2fn(prompt_instructions, prompt_path):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(prompt_path).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["input"], task_dict["topic"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def encode_prompt_s2d(prompt_instructions, prompt_path):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(prompt_path).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["input"], task_dict["topic"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt

def encode_prompt_fn2d(prompt_instructions, prompt_path):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(prompt_path).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["input"], task_dict["topic"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


output_map = {
    'd2s': 'section_text'
}

input_map = {
    'd2s': 'dialogue'
}

def post_process_gpt3_response(num_prompt_instructions, response):
    print("GENERATED OUTPUT:")
    print(response['text'])
    assert False
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./aug-openai/seed.jsonl",
    num_instructions_to_generate=100,
    model_name="text-davinci-003",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    mode='d2s'
):
    assert mode == 'd2s', "sorreh luv, other modes not implemented yet. it does that sometimes."
    encode_prompt = getattr(sys.modules['__main__'], f"encode_prompt_{mode}", False)
    print(encode_prompt)
    # seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = pd.read_csv(seed_tasks_path).to_dict(orient='records')
    
    # seed_instruction_data = [
    #     {"topic": t["section_header"], "output": t['section_text'], "input": t["dialogue"]}
    #     for t in seed_tasks
    # ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    input_field = input_map[mode]
    output_field = output_map[mode]
    all_instructions = [d[output_field] for d in seed_instruction_data] + [
        d[output_field] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        batch_inputs_raw = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt, inputs_raw = encode_prompt(prompt_instructions)
            batch_inputs_raw.append(inputs_raw)
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=512,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=['*END*']
            # stop=["\n20", "20.", "20."],
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            # logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result, (input, topic) in zip(results, batch_inputs_raw):
            new_instructions =  {"topic": topic, input_field: input, output_field: result['text']}# post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data.append(new_instructions)
        print(instruction_data)

        total = len(instruction_data)
        keep = 0

        # TODO: Questionable if we need this
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry[output_field])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry[output_field])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(generate_instruction_following_data)
