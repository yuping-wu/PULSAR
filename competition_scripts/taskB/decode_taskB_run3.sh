echo "Downloading the model..."
wget https://kant.cs.man.ac.uk/data/public/llama-13b.tar.gz
wget https://kant.cs.man.ac.uk/data/public/llama13b-lora-pulsar.tar.gz
echo "Model Downloaded!"

echo "Extracting the model..."
mkdir llama-13b-base
mkdir llama13b-lora-pulsar
tar -xvf llama-13b.tar.gz -C llama-13b-base
tar -xvf llama13b-lora-pulsar.tar.gz -C llama13b-lora-pulsar
echo "Extracted the model!"

echo "Running inference"
python inference_llama.py llama-13b-base llama13b-lora-pulsar $1 --preds_out taskB_PULSAR_run3.csv
echo "Inference completed!"