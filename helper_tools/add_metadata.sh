

runs_folder=$1
metadata_folder=$2
output_folder="with_metadata"

# Check if the output folder exists, if not create it
if [ ! -d $output_folder ]; then
    mkdir $output_folder
fi

# Iterate over all folders in the runs folder storing their name, then use the same name to look up the metadedata md in the metadata folder
for folder in $runs_folder/*; do
    folder_name=$(basename $folder)

    # Check if it is a folder
    if [ ! -d $folder ]; then
        continue
    fi

    metadata_file=$metadata_folder/$folder_name.md
    if [ -f $metadata_file ]; then
        echo "Adding metadata to $folder_name"
        echo "Metadata file: $metadata_file Model json: $folder/$folder_name"_results.json" output folder: $output_folder"
        python helper_tools/include_metadata.py --model_json $folder/$folder_name"_results.json" --metadata_path $metadata_file --out_prefix $output_folder
    else
        echo "No metadata found for $folder_name"
    fi
done