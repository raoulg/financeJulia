# Dataset
dataset comes from this paper plus repo
https://github.com/squareRoot3/Rethinking-Anomaly-Detection

# Optional
The preprocess and save data steps are optional

## preprocess
The preprocess folder contains the requirements.txt file and a jupyter notebook used to transform the original dataformat into csv files.

This assumes the `tfinance` dataset to be in the data/raw folder.

# save data
This transforms the .csv files into a GNNGraph format and saves it as a `.jld2` file.

You can find the result of this in `data/processed/`.
You will need https://git-lfs.github.com/ to extract the file from git.

# Visualize

This loads the `tfinance.jld2` file and creates scatterplot / gif from the t-SNE-pi reduction.

# GNN
## 03_Cora
A GNN model to test with the MNIST for graphs

## 03_graph_NN 
A first setup, adapting the cora script for tfinance

