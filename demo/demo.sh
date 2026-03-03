#!\bin\bash

# A Demo script for rep learning pipeline 

# Download dataset from HF
if [ ! -d "demo/irises_demo" ]; then git clone git@hf.co:datasets/lqmeyers/irises_demo demo/irises_demo; fi

# Run detection, cropping, embedding using demo.yml
python engine.py demo/irises_demo/metadata.csv demo/demo.yml
