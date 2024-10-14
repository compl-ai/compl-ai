# MIT License
#
# Copyright (c) 2021 SoumyaBarikeri
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import re

import torch


def perpelxity_score_simple(sentence, model):

    model.loglikelihood([("", sentence)])[0]

def perplexity_score(sentence, model, tokenizer):
    """
    Finds perplexity score of a sentence based on model
    Parameters
    ----------
    sentence : str
    Given sentence
    model :
    Pre-trained language model
    tokenizer :
    Pre-trained tokenizer

    Returns
    -------
    Perplexity score
    """
    with torch.no_grad():
        model.eval()
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = model(tensor_input, labels=tensor_input)
        return math.exp(loss[0])


def model_perplexity(sentences, model, tokenizer):
    """
    Finds model perplexity based on average model loss over all sentences
    Parameters
    ----------
    sentences : list
    sentence set
    model :
    Pre-trained language model
    tokenizer :
    Pre-trained tokenizer

    Returns
    -------
    Model perplexity score
    """
    total_loss = 0
    for sent in sentences:
        with torch.no_grad():
            model.eval()
            tokenize_input = tokenizer.tokenize(sent)
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
            loss = model(tensor_input, labels=tensor_input)
            total_loss += loss[0]
    return math.exp(total_loss / len(sentences))


def process_tweet(sent):
    """
    Pre-processes a given sentence
    Parameters
    ----------
    sent : str
    Given sentence

    Returns
    -------
    Processed sentence
    """
    sent = sent.encode("ascii", errors="ignore").decode()  # check this output
    # print(sent)
    sent = re.sub("@[^\s]+", "", sent)
    sent = re.sub("https: / /t.co /[^\s]+", "", sent)
    sent = re.sub("http: / /t.co /[^\s]+", "", sent)
    sent = re.sub("http[^\s]+", "", sent)

    # split camel case combined words
    sent = re.sub("([A-Z][a-z]+)", r"\1", re.sub("([A-Z]+)", r" \1", sent))

    sent = sent.lower()

    # remove numbers
    sent = re.sub(" \d+", "", sent)
    # remove words with letter+number
    sent = re.sub("\w+\d+|\d+\w+", "", sent)

    # remove spaces
    sent = re.sub("[\s]+", " ", sent)
    sent = re.sub(r"[^\w\s,.!?]", "", sent)

    # remove 2 or more repeated char
    sent = re.sub(r"(.)\1{2,}", r"\1", sent)
    sent = re.sub(" rt ", "", sent)

    sent = re.sub("- ", "", sent)
    sent = sent.strip()

    # print(sent)
    return sent
