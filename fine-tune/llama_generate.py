import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from callbacks import Iteratorize, Stream
from prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "pulsar_dialogue2note",  # The prompt template to use, will default to alpaca.
    dataset: str = None,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def generate_text(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)
    instruction = "Given this dialogue between a doctor and a patient, generate a note that summarizes the conversation between them."
    if dataset:
        from handystuff.loaders import load_jsonl
        ds = load_jsonl(dataset)
        for e in ds:
            instruction = ...
    else:
        
        input_dialogue = "Doctor: When did your pain begin? Patient: I've had low back pain for about eight years now. Doctor: Is there any injury? Patient: Yeah, it started when I fell in an A B C store. Doctor: How old are you now? Patient: I'm twenty six. Doctor: What kind of treatments have you had for this low back pain? Patient: Yeah, I got referred to P T, and I went, but only once or twice, um, and if I remember right, they only did the electrical stimulation, and heat. Doctor: I see, how has your pain progressed over the last eight years? Patient: It's been pretty continuous, but it's been at varying degrees, sometimes are better than others. Doctor: Do you have any children? Patient: Yes, I had my son in August of two thousand eight, and I've had back pain since giving birth. Doctor: Have you had any falls since the initial one? Patient: Yes, I fell four or five days ago while I was mopping the floor. Doctor: Did you land on your lower back again? Patient: Yes, right onto my tailbone. Doctor: Did that make the low back pain worse? Patient: Yes. Doctor: Have you seen any other doctors for this issue? Patient: Yes, I saw Doctor X on January tenth two thousand nine, and I have a follow up appointment scheduled for February tenth two thousand nine."
        response = generate_text(instruction, input_dialogue)
        print(response)
        return response


if __name__ == "__main__":
    fire.Fire(main)
