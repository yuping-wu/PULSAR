echo "Downloading the model..."
wget https://kant.cs.man.ac.uk/data/public/flan-t5-11b.tar.gz
echo "Model Downloaded!"

echo "Extracting the model..."
mkdir flan-t5-11b
tar -xvf flan-t5-11b.tar.gz -C flan-t5-11b
echo "Extracted the model!"

echo "Running inference"
python inference_t5.py flan-t5-11b $1 taskB_PULSAR_run2.csv
echo "Inference completed!"

