
cat prompts.json | jq -s .[].additional_info.answers | jq -s | jq 'map(select(. != null)) | flatten' > prompt_list.json
