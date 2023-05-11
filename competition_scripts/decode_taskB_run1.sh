echo "Downloading the model..."
wget https://kant.cs.man.ac.uk/data/public/clef-flan-t5-large-bs4-i476-o214-e3-lr3e5-noh.tar.gz
echo "Model Downloaded!"

echo "Extracting the model..."
tar -xvf clef-flan-t5-large-bs4-i476-o214-e3-lr3e5-noh.tar.gz
echo "Extracted the model!"


echo "Running inference"
$MODEL_PATH="clef-flan-t5-large-bs4-i476-o214-e3-lr3e5-noh/2023/epoch2"
python inference_t5.py output/epoch_2_saved $1 taskB_PULSAR_run1.csv
echo "Inference completed!"
