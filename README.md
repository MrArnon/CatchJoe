# Catch Joe Project 

# Installation

## Install Anaconda
For download, Anaconda use the link to the official site https://www.anaconda.com/

## Configurate Anaconda environment
Open the Anaconda Prompt (Terminal with base conda environment). Use the following commands
```bash 
conda deactivate
conda create -n catch_joe python=3.7
conda activate catch_joe
```
## Clone the repository and install requirements
https://github.com/MrArnon/CatchJoe.git

Use activated conda env and move to the project folder.
Execute pip command
```bash 
pip install -r requirements.txt
```

# Usage

To see the data analysis run all cells in data_analysis.ipynb

To run the whole pipeline execute
```bash
conda activate catch_joe
python pipeline.py
```
The predicted result would be stored by path: **./data/predicted_verify.csv** or **./data/predicted_verify.json** 

After changes to fix codestyle run
```bash
make codestyle
```

# Structure
data_analysis.ipynb - data analysis and conclusions for the train data

preprocessing.py - module to preprocess datasets

pipeline.py - script to run a pipeline

config.json - configuration file