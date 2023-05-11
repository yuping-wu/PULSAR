echo "Downloading the model..."
wget https://kant.cs.man.ac.uk/data/public/pulsar-11b.tar.gz
echo "Model Downloaded!"

echo "Extracting the model..."
mkdir pulsar-11b
tar -xvf pulsar-11b.tar.gz -C pulsar-11b
echo "Extracted the model!"

echo "Running inference"
python inference_t5.py pulsar-11b $1 taskB_PULSAR_run1.csv
echo "Inference completed!"
