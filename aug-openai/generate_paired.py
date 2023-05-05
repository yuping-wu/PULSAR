"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
from collections import defaultdict
import sys
import time
import json
from handystuff.loaders import load_jsonl
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
    "fam/sochx": "FAMILY HISTORY/SOCIAL HISTORY",
    "genhx": "HISTORY of PRESENT ILLNESS",
    "pastmedicalhx": "PAST MEDICAL HISTORY",
    "cc": "CHIEF COMPLAINT",
    "pastsurgical": "PAST SURGICAL HISTORY",
    "allergy": None,
    "ros": "REVIEW OF SYSTEMS",
    "medications": None,
    "assessment": None,
    "exam": None,
    "diagnosis": None,
    "disposition": None,
    "plan": None,
    "edcourse": "EMERGENCY DEPARTMENT COURSE",
    "immunizations": None,
    "imaging": None,
    "gynhx": "GYNECOLOGIC HISTORY",
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
        topic = topic_map[topic.lower()] or topic.replace("_", " ")
        topic = topic.lower().capitalize()
        # instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        # input = "<noinput>" if input.lower() == "" else input
        prompt += f"Example Conversation about the patient's {topic}:\n{input}\n"
        prompt += f"Example Output Note:\n{output}\n*END*\n\n"

    input, topic, output = (
        prompt_instructions[-1]["dialogue"],
        prompt_instructions[-1]["section_header"],
        prompt_instructions[-1]["section_text"],
    )
    topic_str = topic_map[topic.lower()] or topic.replace("_", " ")
    topic_str = topic.lower().capitalize()
    prompt += f"Input Conversation about the patient's {topic_str}:\n{input}\n"
    prompt += f"Output Note:"
    return prompt, (input, topic)


def encode_prompt_d2fn(prompt_instructions):
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


def encode_prompt_s2d(prompt_instructions):
    prompt = open("aug-openai/prompt-s2d.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions[:-1]):
        output, topic, input = task_dict["dialogue"], task_dict["section_header"], task_dict["section_text"]
        topic = topic_map[topic.lower()] or topic.replace("_", " ")
        topic = topic.lower().capitalize()
        # instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        # input = "<noinput>" if input.lower() == "" else input
        prompt += f"Example Input Note:\n{input}\n\n"
        prompt += f"Example Conversation about the patient's {topic}:\n{output}\n*END*\n\n"

    topic, input = prompt_instructions[-1]["section_header"], prompt_instructions[-1]["section_text"]
    topic_str = topic_map[topic.lower()] or topic.replace("_", " ")
    topic_str = topic.lower().capitalize()
    prompt += f"Input Note:\n{input}\n\n"
    prompt += f"Output Conversation about the patient's {topic_str}:"
    return prompt, (input, topic)


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


output_map = {"d2s": "section_text", "s2d": "dialogue"}

input_map = {"d2s": "dialogue", "s2d": "section_text"}


non_contributory_terms = ["noncontributory", "non-contributory", "non-contrib"]
def is_noncontributory(datum):
    return any(x in datum['section_text'].lower() for x in non_contributory_terms) or datum['section_text'].strip().lower() == 'nc'

def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./aug-openai/seed.jsonl",
    unlabelled_data_path=None,
    model_name="text-davinci-003",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    mode="d2s",
):
    assert unlabelled_data_path, "This script is for unlabelled augmentation only!"
    assert mode in ["d2s", "s2d"], "sorreh luv, other modes not implemented yet. it does that sometimes."
    encode_prompt = getattr(sys.modules["__main__"], f"encode_prompt_{mode}", False)
    # seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = pd.read_csv(seed_tasks_path).to_dict(orient="records")
    print(output_dir)
    # seed_instruction_data = [
    #     {"topic": t["section_header"], "output": t['section_text'], "input": t["dialogue"]}
    #     for t in seed_tasks
    # ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    if unlabelled_data_path:
        file_modifier = f"{mode}-ul-paired"
        unlabelled_data = load_jsonl(unlabelled_data_path)
        print(f"Loaded {len(unlabelled_data)} unlabelled examples.")
    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, f"regen-{file_modifier}.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, f"regen-{file_modifier}.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")
    print(unlabelled_data_path)

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    
    
    progress_bar = tqdm.tqdm(total=len(unlabelled_data))
    # skip processed examples:
    if machine_instruction_data:
        print(len(machine_instruction_data), 'already annotated!')
        unlabelled_data = unlabelled_data[len(machine_instruction_data):]
        progress_bar.update(len(machine_instruction_data))

      

    # first we tokenize all the seed instructions and generated machine instructions
    input_field = input_map[mode]
    output_field = output_map[mode]
    all_instructions = [d[output_field] for d in seed_instruction_data] + [
        d[output_field] for d in machine_instruction_data
    ]




    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]
    data_by_section = defaultdict(list)
    for d in seed_instruction_data:
        data_by_section[d['section_header'].upper()].append(d)
    while unlabelled_data:
        
        batch_inputs = []
        batch_inputs_raw = []
        for _ in range(request_batch_size):
            if not unlabelled_data:
                break
            ul_datum = unlabelled_data.pop(0)
            # only sampling from the seed tasks
            to_sample_from = data_by_section[ul_datum['section_header'].upper()]
            

            if is_noncontributory(ul_datum):
                to_sample_from = [s for s in to_sample_from if is_noncontributory(s) and len(s['section_text']) < 50]
            if len(to_sample_from) >= num_prompt_instructions - 1:

                prompt_instructions = random.sample(to_sample_from, num_prompt_instructions - 1)
                prompt_instructions.append(ul_datum)

                prompt, inputs_raw = encode_prompt(prompt_instructions)
                batch_inputs_raw.append(inputs_raw)
                batch_inputs.append(prompt)
            else:
                print(f"Can't sample for a datum of type {ul_datum['section_header']}: Nothing left to sample from!")
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=512,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["*END*", "Input", "print", "Output"]
            # stop=["\n20", "20.", "20."],
        )
        request_start = time.time()
        if batch_inputs:
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
                new_instructions = {
                    "topic": topic,
                    input_field: input,
                    output_field: result["text"],
                }  # post_process_gpt3_response(num_prompt_instructions, result)
                instruction_data.append(new_instructions)

            total = len(instruction_data)
            keep = 0
        else:
            instruction_data = []
            print("Batch was empty!")
        if not instruction_data:
            print("Input was censored by openAI due to content management policies. Skipping batch.")
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
        utils.jdump(machine_instruction_data, os.path.join(output_dir, f"regen-{file_modifier}.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(generate_instruction_following_data)
