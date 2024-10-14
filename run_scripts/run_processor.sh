#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No arguments provided. Please provide the parent directory."
    exit 1
fi

# get the parent directory from the first argument
parent_dir=$1

echo "Starting with path: $parent_dir"

# iterate over all subdirectories
for sub_dir in "$parent_dir"/*; do
    # check if it is a directory

    if [ -d "$sub_dir" ]; then
        # extract the base name of the subdirectory
        model_name=$(basename "$sub_dir")

        # print subdirectory and model name
        echo "Processing subdirectory: $sub_dir"
        echo "Model name: $model_name"
        
        # run the python command
        poetry run python3 helper_tools/results_processor.py --parent_dir="$sub_dir" --model_name="$model_name"
        
        # write a message to the console
        echo "Finished processing subdirectory: $model_name"
    fi
done