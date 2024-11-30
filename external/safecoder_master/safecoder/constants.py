FUNC = 0
GOOD = 1
BAD = 2

CWES_TRAINED = [
    "cwe-089",
    "cwe-125",
    "cwe-078",
    "cwe-476",
    "cwe-416",
    "cwe-022",
    "cwe-787",
    "cwe-079",
    "cwe-190",
]

VAL_SCENARIOS = {
    ("cwe-078", "2-py"),
    ("cwe-089", "2-py"),
    ("cwe-125", "2-c"),
    ("cwe-190", "2-c"),
    ("cwe-022", "2-py"),
    ("cwe-787", "2-c"),
}

PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request. Do not repeat the code in already provided in the Response, only append what's necessary to complete the function. Do not put the completion inside ``` brackets. Make sure the code is correctly indented, that is, the first line already starts with 4 spaces as the code is inside a def.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request. Do not repeat the code in already provided in the Response, only append what's necessary to complete the function. Do not put the completion inside ``` brackets. Make sure the code is correctly indented, that is, the first line already starts with 4 spaces as the code is inside a def.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
INSTRUCTION = "Create a {language} function for this problem: {prompt}"

PRETRAINED_MODELS = {
    "codegen-350m": "Salesforce/codegen-350M-multi",
    "codegen-2b": "Salesforce/codegen-2B-multi",
    "starcoderbase-1b": "bigcode/starcoderbase-1b",
    "starcoderbase-3b": "bigcode/starcoderbase-3b",
    "starcoderbase-16b": "bigcode/starcoderbase",
}
