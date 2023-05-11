#!/bin/bash
echo "Starting installation..."

# create and activate the virtual environment
echo "Creating virtual enviroment..."
python -m venv PULSAR_taskB_venv
echo "Virtual environment created!"


# activate the virtual environment
echo "Activating the virtual environment..."
source PULSAR_taskB_venv/bin/activate
echo "Virtual environment activated!"

# install the requirements
echo "Installing the python packages from the requirements file..."
pip install -r requirements.txt
echo "All packages installed!"

# deactivate the virtual environment
echo "Deactivating from the virtual environment and exiting!"
deactivate
