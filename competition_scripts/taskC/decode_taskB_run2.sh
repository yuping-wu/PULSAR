echo "Downloading the model..."
wget https://kant.cs.man.ac.uk/data/public/t5-3b-daug-taskc.tar.gz
echo "Model Downloaded!"

echo "Extracting the model..."
mkdir t5-3b-daug
tar -xvf t5-3b-daug-taskc.tar.gz -C t5-3b-daug
echo "Extracted the model!"

echo "Running inference"
python inference_t5.py t5-3b-daug $1 taskC_PULSAR_run2.csv
echo "Inference completed!"
exit
