echo "Downloading the model..."
wget https://kant.cs.man.ac.uk/data/public/pulsar-11b.tar.gz
echo "Model Downloaded!"

echo "Extracting the model..."
tar -xvf pulsar-11b.tar.gz
echo "Extracted the model!"


echo "Running inference"
$MODEL_PATH="pulsar-11b/"
python inference_t5.py $MODEL_PATH $1 taskB_PULSAR_run1.csv
echo "Inference completed!"
