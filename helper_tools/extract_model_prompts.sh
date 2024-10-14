
cat prompts.json | jq -s .[].prompt | jq -s | jq 'map(select(. != null)) | flatten' > input_list.json
