cat prompts.json | jq -s .[].additional_info.loglikelihoods | jq -s | jq 'map(select(. != null)) | flatten(1)' > prompt_list.json
