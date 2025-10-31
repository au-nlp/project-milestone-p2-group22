#! /bin/bash

# Recreate the conda environment from the env.yaml file
# deletes the existing environment if it exists, by using --force flag

conda env create -f env.yaml --force
