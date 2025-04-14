# AI2-THOR RERUN Visualization

This is an example code to visualize the AI2-THOR environment in RERUN.

## Setting Up the Environment

To get started, create a Conda environment and install the required dependencies:

```bash
# Create a new Conda environment
conda create -n ai2thor_env python=3.8 -y

# Activate the environment
conda activate ai2thor_env

# Install required packages
pip install ai2thor rerun-sdk numpy pillow scipy
```


```bash
# Run the data collector script
python data_collector.py
# Run the visualization script
python rerun_ai2thor.py
```