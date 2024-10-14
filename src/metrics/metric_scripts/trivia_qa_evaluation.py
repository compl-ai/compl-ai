#    Copyright 2017 Kenton Lee
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#    Source: https://github.com/mandarjoshi90/triviaqa/blob/master/utils/utils.py


import json
import re
import string
import sys
from collections import Counter
from typing import Optional


def write_json_to_file(json_object, json_file, mode="w", encoding="utf-8"):
    with open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def get_file_contents(filename, encoding="utf-8"):
    with open(filename, encoding=encoding) as f:
        content = f.read()
    return content


def read_json(filename, encoding="utf-8"):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_file_contents_as_list(file_path, encoding="utf-8", ignore_blanks=True):
    contents = get_file_contents(file_path, encoding=encoding)
    lines = contents.split("\n")
    lines = [line for line in lines if line != ""] if ignore_blanks else lines
    return lines


#  Code taken from https://github.com/mandarjoshi90/triviaqa/blob/master/utils/dataset_utils.py

# Key for wikipedia eval is question-id. Key for web eval is the (question_id, filename) tuple


def get_key_to_ground_truth(data):
    """Get a dictionary of question id to answer for the given data."""

    if data["Domain"] == "Wikipedia":
        return {datum["QuestionId"]: datum["Answer"] for datum in data["Data"]}
    else:
        return get_qd_to_answer(data)


def get_question_doc_string(qid, doc_name):
    """Get a string that represents the question and document."""

    return "{}--{}".format(qid, doc_name)


def get_qd_to_answer(data):
    """Get a dictionary of (question_id, doc_name) to answer for the given data."""

    key_to_answer = {}
    for datum in data["Data"]:
        for page in datum.get("EntityPages", []) + datum.get("SearchResults", []):
            qd_tuple = get_question_doc_string(datum["QuestionId"], page["Filename"])
            key_to_answer[qd_tuple] = datum["Answer"]
    return key_to_answer


def read_clean_part(datum):
    """
    Read only the clean part of the data.
    Clean means that the document is a part of the verified evaluation set.
    Moreover, there has to be at least one answer in the document.
    """

    for key in ["EntityPages", "SearchResults"]:
        new_page_list = []
        for page in datum.get(key, []):
            if page["DocPartOfVerifiedEval"]:
                new_page_list.append(page)
        datum[key] = new_page_list
    assert len(datum["EntityPages"]) + len(datum["SearchResults"]) > 0
    return datum


def read_triviaqa_data(qajson):
    """Read the TriviaQA data from the given file."""

    data = read_json(qajson)
    # read only documents and questions that are a part of clean data set
    if data["VerifiedEval"]:
        clean_data = []
        for datum in data["Data"]:
            if datum["QuestionPartOfVerifiedEval"]:
                if data["Domain"] == "Web":
                    datum = read_clean_part(datum)
                clean_data.append(datum)
        data["Data"] = clean_data
    return data


def answer_index_in_document(answer, document):
    """Get the index of the answer in the document."""

    answer_list = answer["NormalizedAliases"]
    for answer_string_in_doc in answer_list:
        index = document.lower().find(answer_string_in_doc)
        if index != -1:
            return answer_string_in_doc, index
    return answer["NormalizedValue"], -1


# Code from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
# -*- coding: utf-8 -*-
""" Official evaluation script for v1.0 of the TriviaQA dataset.
Extended from the evaluation script for v1.1 of the SQuAD dataset. """


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is an exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute the maximum metric between the prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def is_exact_match(answer_object, prediction):
    """Check if the prediction is an exact match with the ground truth up to normalization."""
    ground_truths = get_ground_truths(answer_object)
    for ground_truth in ground_truths:
        if exact_match_score(prediction, ground_truth):
            return True
    return False


def has_exact_match(ground_truths, candidates):
    """
    Check if there is an exact match between a sample from the ground truth and the candidate.
    """

    for ground_truth in ground_truths:
        if ground_truth in candidates:
            return True
    return False


def get_ground_truths(answer):
    """Get the list of ground truths for the given answer."""

    return answer["normalized_aliases"] + [
        normalize_answer(ans) for ans in answer.get("human_answers", [])
    ]


def get_oracle_score(ground_truth, predicted_answers, qid_list=None, mute=False):
    exact_match = common = 0.0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = "Irrelavant question {} will receive score 0.".format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = normalize_answer(predicted_answers[qid])
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = has_exact_match(ground_truths, prediction)
        exact_match += float(em_for_this_question)

    exact_match = 100.0 * exact_match / len(qid_list)

    return {
        "oracle_exact_match": exact_match,
        "common": common,
        "denominator": len(qid_list),
        "pred_len": len(predicted_answers),
        "gold_len": len(ground_truth),
    }


def evaluate_triviaqa(ground_truth, predicted_answers, qid_list=None, mute=False):
    f1 = exact_match = common = 0.0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = "Missed question {} will receive score 0.".format(qid)
                print(message, file=sys.stderr)
            continue
        if qid not in ground_truth:
            if not mute:
                message = "Irrelavant question {} will receive score 0.".format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = predicted_answers[qid]
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        if em_for_this_question == 0 and not mute:
            print("em=0:", prediction, ground_truths)
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        f1 += f1_for_this_question

    exact_match = 100.0 * exact_match / len(qid_list)
    f1 = 100.0 * f1 / len(qid_list)

    return {
        "exact_match": exact_match,
        "f1": f1,
        "common": common,
        "denominator": len(qid_list),
        "pred_len": len(predicted_answers),
        "gold_len": len(ground_truth),
    }


class TriviaQAEval:
    def _compute_trivia_qa_eval(self, predictions: dict, reference: dict):
        dataset_json = reference
        expected_version = 1.0
        if dataset_json["Version"] != expected_version:
            print(
                "Evaluation expects v-{} , but got dataset with v-{}".format(
                    expected_version, dataset_json["Version"]
                ),
                file=sys.stderr,
            )
        key_to_ground_truth = get_key_to_ground_truth(dataset_json)
        eval_dict = evaluate_triviaqa(key_to_ground_truth, predictions)
        return eval_dict

    """
    def _info(self):
        return datasets.MetricInfo(
            citation=
                    @misc{joshi2017triviaqa,
                    title={TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
                    author={Mandar Joshi and Eunsol Choi and Daniel S. Weld and Luke Zettlemoyer},
                    year={2017},
                    eprint={1705.03551},
                    archivePrefix={arXiv},
                    primaryClass={cs.CL}
                    }
                    ,
            description="normalized total probability for correct choices",
            inputs_description="Both predictions and references are dicts presented as in https://github.com/mandarjoshi90/triviaqa/tree/master/samples",
            features=datasets.Features(
                {
                    "predictions": Any,
                    "references": Any,
                }
            ),
        )
    """

    def compute_result(self):
        pass

    def add_batch(self, predictions: Optional[dict] = None, references: Optional[dict] = None):
        raise NotImplementedError

    def compute(self, predictions: Optional[dict] = None, references: Optional[dict] = None):
        if not predictions or not references:
            raise ValueError("Required arguments predictiosna nd references not present!")
        return {"evaluate_trivia_qa": self._compute_trivia_qa_eval(predictions, references)}

    def evaluate(self, *args, **kwargs) -> float:
        return self.compute(*args, **kwargs)
