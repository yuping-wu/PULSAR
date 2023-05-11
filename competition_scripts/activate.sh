#!/bin/bash
script_dir=`dirname $0`
cd $script_dir
/bin/bash -c ". ./PULSAR_taskB_venv/bin/activate; exec /bin/bash -i"
