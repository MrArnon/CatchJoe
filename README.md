# Catch Joe Project 

# The task
Download the dataset here: https://drive.google.com/file/d/1uXmW_13lf2e_lEpT6K9heIp_zhLHCaOD/view?usp=sharing
 
The dataset contains data about user sessions that have been recorded over a period of time. The dataset consists of two parts: the training dataset where user ID’s are labeled, and the verification set without labels.

Each session is represented by a JSON object with the following fields:
> - *user_id* is the unique identifier of the user.
> - *browser*, *os*, *locale* contain info about the software on the user’s machine.
> - *gender*, *location* give analytics data about the user.
> - *date* and *time* is the moment when the session started (in GMT).
> - *sites* is a list of up to 15 sites visited during the session. For each site, the url and the length of visit in seconds are given.

The goal is to create a method to identify the user with id=0 (codename Joe) specifically.

Your solution should contain:
> - Exploratory data analysis, either as a standalone report/presentation or in the form of a Jupyter notebook
> - A standalone script that runs the whole pipeline on the verification set and creates a file where each line is the predicted label (0 = Joe, 1 = not Joe)

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

# Docker
```bash
docker build -t catch_joe .
docker run -ti --rm -v /Users/maksimpolakov/CatchJoe:/home/catchjoe/app catch_joe python pipeline.py
```

