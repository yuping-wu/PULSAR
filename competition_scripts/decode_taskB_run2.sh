echo "Downloading the model..."
wget https://kant.cs.man.ac.uk/data/public/flan-t5-11b.tar.gz
echo "Model Downloaded!"

echo "Extracting the model..."
tar -xvf flan-t5-11b.tar.gz
echo "Extracted the model!"


echo "Running inference"
$MODEL_PATH="flan-t5-11b"
python inference_t5.py $MODEL_PATH $1 taskB_PULSAR_run2.csv
echo "Inference completed!"
