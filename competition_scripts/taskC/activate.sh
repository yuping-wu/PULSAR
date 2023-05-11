#!/bin/bash
# We are using a subshell to activate the virtual environment and persist changes.
# Since the competition instructions mentioned the file would be run as ./activate.sh, the virtual environment is activated with the shell process created by running it, but the activation of the virtual environment does not
# persist once the shell file finishes running
# This script gives rise to a subshell with the virtualenv invoked, and the other scripts can be run normally
# To exit from the subshell, please use the exit command
# Script referred from https://stackoverflow.com/a/13123926

script_dir=`dirname $0`
cd $script_dir
/bin/bash -c ". ./PULSAR_taskC_venv/bin/activate; exec /bin/bash -i"
