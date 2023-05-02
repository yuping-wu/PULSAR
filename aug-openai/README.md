# Data Aug with Azure OpenAI
Again, assuming `../clef-medqa` is the location of the CLEF data.

```bash
thon aug-openai/generate.py --seed-tasks-path=../clef-medqa/dataset/TaskB/TaskB-TrainingSet.csv --mode d2s --num_prompt_instructions 3 --top_p=0.5 --request_batch_size 1 --num_instructions_to_generate 35
```
is an example of how to run the script.


`--num_prompt_instructions` is how many examples to feed before generating output.

Here, we generate only one example per API call. I think reducing the number of prompt instructions should make things cheaper.

So far, only `--mode d2s` is implemented (i.e. given dialogue, generate note). Need to generate `s2d` and `d2fn` and `fn2d` (for `x2d`: use mimic.)