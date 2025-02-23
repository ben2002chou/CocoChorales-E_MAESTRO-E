# Generating CocoChorales-E and MAESTRO-E

// --- Overview ---
This guide instructs you on generating the CocoChorales-E and MAESTRO-E datasets. 

# Setup:
It is suggested to run these commands in this order to avoid potential package conflicts due to midi-ddsp having outdated dependancies from tensorflow.

## Environment Setup:
```bash
conda create -n your-env python=3.8
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
pip install midi-ddsp
pip install numpy==1.23.0
pip install pyloudnorm
```

## Model Weights & Additional Tools:
```bash
# Run the jupyter notebook which downloads a few files:
# in midi-ddsp github:
python download_model_weights.py 
conda install conda-forge::fluidsynth
pip install pyFluidSynth
```

# How to run:
First, download the CocoChorales and MAESTRO datasets into your own [dataset_folder].

```bash
conda activate your-env
cd [project_folder]
```

## CocoChorales-E:
```bash
base_data_dir=[dataset_folder]/cocochorales_full/org_chunked_midi/random 
output_dir=[output_dir]

mkdir -p $output_dir

for i in {0..63}; do
    data_dir="${base_data_dir}/${i}"
    echo "Processing directory: ${data_dir}"
    python create_cocochorales.py --midi_dir ${data_dir} --output_dir ${output_dir}
done
```

## MAESTRO-E:
```bash
data_dir=[dataset_folder]/maestro/maestro-v3.0.0
output_dir=[output_dir]

mkdir -p $output_dir

python create_maestro.py --midi_dir ${data_dir} --output_dir ${output_dir}