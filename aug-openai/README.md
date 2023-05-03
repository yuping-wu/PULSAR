# Data Aug with Azure OpenAI
Again, assuming `../clef-medqa` is the location of the CLEF data.

```bash
thon aug-openai/generate.py --seed-tasks-path=../clef-medqa/dataset/TaskB/TaskB-TrainingSet.csv --mode d2s --num_prompt_instructions 3 --top_p=0.5 --request_batch_size 1 --num_instructions_to_generate 35
```
is an example of how to run the script.


`--num_prompt_instructions` is how many examples to feed before generating output.

Here, we generate only one example per API call. I think reducing the number of prompt instructions should make things cheaper.

So far, only `--mode d2s` and `s2d` are implemented (i.e. given dialogue, generate note and vice versa). Need to generate `d2fn` and `fn2d`.

To annotate unlabelled examples in `s2d`, i.e. section to dialogue, use the following command.

```bash
python aug-openai/generate.py --num_instructions_to_generate 2 --request_batch_size 1 --num_prompt_instructions 2 --top_p 0.5 --mode s2d --seed_tasks_path ../clef-medqa/dataset/TaskB/TaskB-TrainingSet.csv --unlabelled_data_path aug-openai/unlabelled.jsonl
```

To obtain a bunch of sections extracted from mimic (so far, unprocessed), install git-lfs and unzip the zipped tarball `aug-openai/unlabelled.tar.gz`.