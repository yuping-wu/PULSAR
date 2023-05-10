echo "Downloading the model..."
wget https://kant.cs.man.ac.uk/data/public/clef-flan-t5-large-bs4-i476-o214-e3-lr3e5-noh.tar.gz
echo "Model Downloaded!"

echo "Extracting the model..."
tar -xvf clef-flan-t5-large-bs4-i476-o214-e3-lr3e5-noh.tar.gz
echo "Extracted the model!"


echo "Running inference"
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file finetune_accelerate_fsdp.yml clef_finetune_accelerate.py\
 --model_path clef-flan-t5-large-bs4-i476-o214-e3-lr3e5-noh/2023/epoch_2/ --model_name google/flan-t5-large \
 --tokenizer_name google/flan-t5-large --input_file $1 --input_test_file $1 --output_dir output_clef\
 --output_file system.txt --max_input_length 512 --per_device_eval_batch_size 4  --test
echo "Inference completed!"
