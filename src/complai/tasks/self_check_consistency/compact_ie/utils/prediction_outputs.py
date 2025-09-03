import json
from collections import defaultdict


def read_conjunctions(cfg):
    conj2sent = dict()
    file_path = cfg.conjunctions_file

    with open(file_path, "r") as fin:
        sent = 1
        currentSentText = ""
        for line in fin:
            if line == "\n":
                sent = 1
                continue
            elif sent == 1:
                currentSentText = line.replace("\n", "")
                sent = 0
            else:
                conj_sent = line.replace("\n", "")
                conj2sent[conj_sent] = currentSentText
    conj_sentences = list(conj2sent.keys())
    return conj_sentences, conj2sent


def print_predictions(outputs, file_path, vocab, sequence_label_domain=None):
    """print_predictions prints prediction results

    Args:
        outputs (list): prediction outputs
        file_path (str): output file path
        vocab (Vocabulary): vocabulary
        sequence_label_domain (str, optional): sequence label domain. Defaults to None.
    """
    with open(file_path, "w") as fout:
        for sent_output in outputs:
            seq_len = sent_output["seq_len"]
            assert "tokens" in sent_output
            tokens = [
                vocab.get_token_from_index(token, "tokens")
                for token in sent_output["tokens"][:seq_len]
            ]
            print("Token\t{}".format(" ".join(tokens)), file=fout)

            if "text" in sent_output:
                print(f"Text\t{sent_output['text']}", file=fout)

            if (
                "sequence_labels" in sent_output
                and "sequence_label_preds" in sent_output
            ):
                sequence_labels = [
                    vocab.get_token_from_index(
                        true_sequence_label, sequence_label_domain
                    )
                    for true_sequence_label in sent_output["sequence_labels"][:seq_len]
                ]
                sequence_label_preds = [
                    vocab.get_token_from_index(
                        pred_sequence_label, sequence_label_domain
                    )
                    for pred_sequence_label in sent_output["sequence_label_preds"][
                        :seq_len
                    ]
                ]

                print(
                    "Sequence-Label-True\t{}".format(" ".join(sequence_labels)),
                    file=fout,
                )
                print(
                    "Sequence-Label-Pred\t{}".format(" ".join(sequence_label_preds)),
                    file=fout,
                )

            if "joint_label_matrix" in sent_output:
                for row in sent_output["joint_label_matrix"][:seq_len]:
                    print(
                        "Joint-Label-True\t{}".format(
                            " ".join(
                                [
                                    vocab.get_token_from_index(item, "ent_rel_id")
                                    for item in row[:seq_len]
                                ]
                            )
                        ),
                        file=fout,
                    )

            if "joint_label_preds" in sent_output:
                for row in sent_output["joint_label_preds"][:seq_len]:
                    print(
                        "Joint-Label-Pred\t{}".format(
                            " ".join(
                                [
                                    vocab.get_token_from_index(item, "ent_rel_id")
                                    for item in row[:seq_len]
                                ]
                            )
                        ),
                        file=fout,
                    )

            if "separate_positions" in sent_output:
                print(
                    "Separate-Position-True\t{}".format(
                        " ".join(map(str, sent_output["separate_positions"]))
                    ),
                    file=fout,
                )

            if "all_separate_position_preds" in sent_output:
                print(
                    "Separate-Position-Pred\t{}".format(
                        " ".join(map(str, sent_output["all_separate_position_preds"]))
                    ),
                    file=fout,
                )

            if "span2ent" in sent_output:
                for span, ent in sent_output["span2ent"].items():
                    ent = vocab.get_token_from_index(ent, "span2ent")
                    assert ent != "None", "true relation can not be `None`."

                    print(
                        "Ent-True\t{}\t{}\t{}".format(
                            ent, span, " ".join(tokens[span[0] : span[1]])
                        ),
                        file=fout,
                    )

            if "all_ent_preds" in sent_output:
                for span, ent in sent_output["all_ent_preds"].items():
                    # ent = vocab.get_token_from_index(ent, 'span2ent')

                    print("Ent-Span-Pred\t{}".format(span), file=fout)
                    print(
                        "Ent-Pred\t{}\t{}\t{}".format(
                            ent, span, " ".join(tokens[span[0] : span[1]])
                        ),
                        file=fout,
                    )

            if "span2rel" in sent_output:
                for (span1, span2), rel in sent_output["span2rel"].items():
                    rel = vocab.get_token_from_index(rel, "span2rel")
                    assert rel != "None", "true relation can not be `None`."

                    if rel[-1] == "<":
                        span1, span2 = span2, span1
                    print(
                        "Rel-True\t{}\t{}\t{}\t{}\t{}".format(
                            rel[:-2],
                            span1,
                            span2,
                            " ".join(tokens[span1[0] : span1[1]]),
                            " ".join(tokens[span2[0] : span2[1]]),
                        ),
                        file=fout,
                    )

            if "all_rel_preds" in sent_output:
                for (span1, span2), rel in sent_output["all_rel_preds"].items():
                    # rel = vocab.get_token_from_index(rel, 'span2rel')

                    if rel[-1] == "<":
                        span1, span2 = span2, span1
                    print(
                        "Rel-Pred\t{}\t{}\t{}\t{}\t{}".format(
                            rel[:-2],
                            span1,
                            span2,
                            " ".join(tokens[span1[0] : span1[1]]),
                            " ".join(tokens[span2[0] : span2[1]]),
                        ),
                        file=fout,
                    )

            print(file=fout)


def print_extractions_allennlp_format(cfg, outputs, file_path, vocab):
    conj_sentences, conj2sent = read_conjunctions(cfg)
    ext_texts = []
    with open(file_path, "w") as fout:
        for sent_output in outputs:
            extractions = {}
            seq_len = sent_output["seq_len"]
            assert "tokens" in sent_output
            tokens = [
                vocab.get_token_from_index(token, "tokens")
                for token in sent_output["tokens"][: seq_len - 6]
            ]
            sentence = " ".join(tokens)
            if sentence in conj_sentences:
                sentence = conj2sent[sentence]

            if "all_rel_preds" in sent_output:
                for (span1, span2), rel in sent_output["all_rel_preds"].items():
                    if rel == "" or rel == " ":
                        continue
                    if sent_output["all_ent_preds"][span1] == "Relation":
                        try:
                            if span2 in extractions[span1][rel]:
                                continue
                        except:
                            pass
                        try:
                            extractions[span1][rel].append(span2)
                        except:
                            extractions[span1] = defaultdict(list)
                            extractions[span1][rel].append(span2)
                    else:
                        try:
                            if span1 in extractions[span2][rel]:
                                continue
                        except:
                            pass
                        try:
                            extractions[span2][rel].append(span1)
                        except:
                            extractions[span2] = defaultdict(list)
                            extractions[span2][rel].append(span1)
            to_remove_rel_spans = set()
            expand_rel = {}
            to_add = {}
            for rel_span1, d1 in extractions.items():
                for rel_span2, d2 in extractions.items():
                    if rel_span1 != rel_span2 and not (
                        rel_span1 in to_remove_rel_spans
                        or rel_span2 in to_remove_rel_spans
                    ):
                        if (
                            d1["Subject"] == d2["Subject"]
                            and d1["Object"] == d2["Object"]
                        ):
                            if rel_span1 in to_remove_rel_spans:
                                to_add[expand_rel[rel_span1] + rel_span2] = d1
                                to_remove_rel_spans.add(rel_span2)
                                to_remove_rel_spans.add(expand_rel[rel_span1])
                                expand_rel[rel_span2] = (
                                    expand_rel[rel_span1] + rel_span2
                                )
                                expand_rel[rel_span1] = (
                                    expand_rel[rel_span1] + rel_span2
                                )
                            elif rel_span2 in to_remove_rel_spans:
                                to_add[expand_rel[rel_span2] + rel_span1] = d1
                                to_remove_rel_spans.add(rel_span1)
                                to_remove_rel_spans.add(expand_rel[rel_span2])
                                expand_rel[rel_span1] = (
                                    expand_rel[rel_span2] + rel_span1
                                )
                                expand_rel[rel_span2] = (
                                    expand_rel[rel_span2] + rel_span1
                                )
                            else:
                                to_add[rel_span1 + rel_span2] = d1
                                expand_rel[rel_span1] = rel_span1 + rel_span2
                                expand_rel[rel_span2] = rel_span1 + rel_span2
                                to_remove_rel_spans.add(rel_span1)
                                to_remove_rel_spans.add(rel_span2)
            for tm in to_remove_rel_spans:
                del extractions[tm]
            for k, v in to_add.items():
                extractions[k] = v
            for rel_sp, d in extractions.items():
                if len(d["Subject"]) > 1:
                    sorted_d_subject = sorted(d["Subject"], key=lambda x: x[0][0])
                    sorted_d_subject = [x[0] for x in sorted_d_subject]
                    subject_text = " ".join(
                        [
                            " ".join(tokens[sub_span[0] : sub_span[1]])
                            for sub_span in sorted_d_subject
                        ]
                    )
                elif len(d["Subject"]) == 1:
                    subject_text = " ".join(
                        [
                            " ".join(tokens[sub_span[0] : sub_span[1]])
                            for sub_span in d["Subject"][0]
                        ]
                    )
                else:
                    subject_text = ""
                if len(d["Object"]) > 1:
                    sorted_d_object = sorted(d["Object"], key=lambda x: x[0][0])
                    sorted_d_object = [x[0] for x in sorted_d_object]
                    object_text = " ".join(
                        [
                            " ".join(tokens[sub_span[0] : sub_span[1]])
                            for sub_span in sorted_d_object
                        ]
                    )
                elif len(d["Object"]) == 1:
                    object_text = " ".join(
                        [
                            " ".join(tokens[sub_span[0] : sub_span[1]])
                            for sub_span in d["Object"][0]
                        ]
                    )
                else:
                    object_text = ""
                rel_text = " ".join(
                    [" ".join(tokens[sub_span[0] : sub_span[1]]) for sub_span in rel_sp]
                ).replace("[unused1]", "is")
                ext = f"<arg1> {subject_text} </arg1> <rel> {rel_text} </rel> <arg2> {object_text} </arg2>"
                if ext not in ext_texts and (rel_text != "" and subject_text != ""):
                    print("{}\t{}".format(sentence, ext), file=fout)
                ext_texts.append(ext)


def print_predictions_for_joint_decoding(outputs, file_path, vocab):
    """print_predictions prints prediction results

    Args:
        outputs (list): prediction outputs
        file_path (str): output file path
        vocab (Vocabulary): vocabulary
        sequence_label_domain (str, optional): sequence label domain. Defaults to None.
    """
    with open(file_path, "w") as fout:
        for sent_output in outputs:
            seq_len = sent_output["seq_len"]
            assert "tokens" in sent_output
            tokens = [
                vocab.get_token_from_index(token, "tokens")
                for token in sent_output["tokens"][:seq_len]
            ]
            print("Token\t{}".format(" ".join(tokens)), file=fout)

            if "joint_label_matrix" in sent_output:
                for row in sent_output["joint_label_matrix"][:seq_len]:
                    print(
                        "Joint-Label-True\t{}".format(
                            " ".join(
                                [
                                    vocab.get_token_from_index(item, "ent_rel_id")
                                    for item in row[:seq_len]
                                ]
                            )
                        ),
                        file=fout,
                    )

            if "joint_label_preds" in sent_output:
                for row in sent_output["joint_label_preds"][:seq_len]:
                    print(
                        "Joint-Label-Pred\t{}".format(
                            " ".join(
                                [
                                    vocab.get_token_from_index(item, "ent_rel_id")
                                    for item in row[:seq_len]
                                ]
                            )
                        ),
                        file=fout,
                    )

            if "separate_positions" in sent_output:
                print(
                    "Separate-Position-True\t{}".format(
                        " ".join(map(str, sent_output["separate_positions"]))
                    ),
                    file=fout,
                )

            if "all_separate_position_preds" in sent_output:
                print(
                    "Separate-Position-Pred\t{}".format(
                        " ".join(map(str, sent_output["all_separate_position_preds"]))
                    ),
                    file=fout,
                )

            if "all_ent_span_preds" in sent_output:
                for span in sent_output["all_ent_span_preds"]:
                    print("Ent-Span-Pred\t{}".format(span), file=fout)

            if "span2ent" in sent_output:
                for span, ent in sent_output["span2ent"].items():
                    ent = vocab.get_token_from_index(ent, "ent_rel_id")
                    assert ent != "None", "true relation can not be `None`."

                    print(
                        "Ent-True\t{}\t{}\t{}".format(
                            ent,
                            span,
                            " ".join(
                                [
                                    " ".join(tokens[sub_span[0] : sub_span[1]])
                                    for sub_span in span
                                ]
                            ),
                        ),
                        file=fout,
                    )

            if "all_ent_preds" in sent_output:
                for span, ent in sent_output["all_ent_preds"].items():
                    # ent = vocab.get_token_from_index(ent, 'span2ent')
                    print(
                        "Ent-Pred\t{}\t{}\t{}".format(
                            ent,
                            span,
                            " ".join(
                                [
                                    " ".join(tokens[sub_span[0] : sub_span[1]])
                                    for sub_span in span
                                ]
                            ),
                        ),
                        file=fout,
                    )

            if "span2rel" in sent_output:
                for (span1, span2), rel in sent_output["span2rel"].items():
                    rel = vocab.get_token_from_index(rel, "ent_rel_id")
                    assert rel != "None", "true relation can not be `None`."
                    span1_text_list = [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in span1
                    ]
                    span2_text_list = [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in span2
                    ]
                    print(
                        "Rel-True\t{}\t{}\t{}\t{}\t{}".format(
                            rel,
                            span1,
                            span2,
                            " ".join(span1_text_list),
                            " ".join(span2_text_list),
                        ),
                        file=fout,
                    )

            if "all_rel_preds" in sent_output:
                for (span1, span2), rel in sent_output["all_rel_preds"].items():
                    # rel = vocab.get_token_from_index(rel, 'span2rel')

                    span1_text_list = [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in span1
                    ]
                    span2_text_list = [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in span2
                    ]
                    print(
                        "Rel-Pred\t{}\t{}\t{}\t{}\t{}".format(
                            rel,
                            span1,
                            span2,
                            " ".join(span1_text_list),
                            " ".join(span2_text_list),
                        ),
                        file=fout,
                    )

                    # print("Rel-Pred\t{}\t{}\t{}\t{}\t{}".format(rel, span1, span2, ' '.join(tokens[span1[0]:span1[1]]),
                    #                                             ' '.join(tokens[span2[0]:span2[1]])),
                    #       file=fout)

            print(file=fout)


def print_predictions_for_entity_rel_decoding(outputs, file_path, vocab):
    """print_predictions prints prediction results

    Args:
        outputs (list): prediction outputs
        file_path (str): output file path
        vocab (Vocabulary): vocabulary
        sequence_label_domain (str, optional): sequence label domain. Defaults to None.
    """
    with open(file_path, "w") as fout:
        # for sent_output, rel_sent_output in zip(outputs, rel_outputs):
        for sent_output in outputs:
            seq_len = sent_output["seq_len"]
            assert "tokens" in sent_output
            tokens = [
                vocab.get_token_from_index(token, "tokens")
                for token in sent_output["tokens"][:seq_len]
            ]
            print("Token\t{}".format(" ".join(tokens)), file=fout)

            if "entity_label_preds" in sent_output:
                for row in sent_output["entity_label_preds"][:seq_len]:
                    print(
                        "Ent-Label-Pred\t{}".format(
                            " ".join(
                                [
                                    vocab.get_token_from_index(item, "ent_rel_id")
                                    for item in row[:seq_len]
                                ]
                            )
                        ),
                        file=fout,
                    )

            if "relation_label_matrix" in sent_output:
                for row in sent_output["relation_label_matrix"][:seq_len]:
                    print(
                        "Rel-Label-True\t{}".format(
                            " ".join(
                                [
                                    vocab.get_token_from_index(item + 2, "ent_rel_id")
                                    if item != 0
                                    else "None"
                                    for item in row[:seq_len]
                                ]
                            )
                        ),
                        file=fout,
                    )

            if "relation_label_preds" in sent_output:
                for row in sent_output["relation_label_preds"][:seq_len]:
                    print(
                        "Rel-Label-Pred\t{}".format(
                            " ".join(
                                [
                                    vocab.get_token_from_index(item + 2, "ent_rel_id")
                                    if item != 0
                                    else "None"
                                    for item in row[:seq_len]
                                ]
                            )
                        ),
                        file=fout,
                    )

            if "separate_positions" in sent_output:
                print(
                    "Separate-Position-True\t{}".format(
                        " ".join(map(str, sent_output["separate_positions"]))
                    ),
                    file=fout,
                )

            if "all_separate_position_preds" in sent_output:
                print(
                    "Separate-Position-Pred\t{}".format(
                        " ".join(map(str, sent_output["all_separate_position_preds"]))
                    ),
                    file=fout,
                )

            if "all_ent_span_preds" in sent_output:
                for span in sent_output["all_ent_span_preds"]:
                    print("Ent-Span-Pred\t{}".format(span), file=fout)

            if "span2ent" in sent_output:
                for span, ent in sent_output["span2ent"].items():
                    ent = vocab.get_token_from_index(ent, "ent_rel_id")
                    assert ent != "None", "true relation can not be `None`."

                    print(
                        "Ent-True\t{}\t{}\t{}".format(
                            ent,
                            span,
                            " ".join(
                                [
                                    " ".join(tokens[sub_span[0] : sub_span[1]])
                                    for sub_span in span
                                ]
                            ),
                        ),
                        file=fout,
                    )

            if "all_ent_preds" in sent_output:
                for span, ent in sent_output["all_ent_preds"].items():
                    print(
                        "Ent-Pred\t{}\t{}\t{}".format(
                            ent,
                            span,
                            " ".join(
                                [
                                    " ".join(tokens[sub_span[0] : sub_span[1]])
                                    for sub_span in span
                                ]
                            ),
                        ),
                        file=fout,
                    )

            if "span2rel" in sent_output:
                for (span1, span2), rel in sent_output["span2rel"].items():
                    rel = vocab.get_token_from_index(rel, "ent_rel_id")
                    assert rel != "None", "true relation can not be `None`."

                    span1_text_list = [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in span1
                    ]
                    span2_text_list = [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in span2
                    ]
                    print(
                        "Rel-True\t{}\t{}\t{}\t{}\t{}".format(
                            rel,
                            span1,
                            span2,
                            " ".join(span1_text_list),
                            " ".join(span2_text_list),
                        ),
                        file=fout,
                    )
            if "all_rel_preds" in sent_output:
                for (span1, span2), rel in sent_output["all_rel_preds"].items():
                    span1_text_list = [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in span1
                    ]
                    span2_text_list = [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in span2
                    ]
                    print(
                        "Rel-Pred\t{}\t{}\t{}\t{}\t{}".format(
                            rel,
                            span1,
                            span2,
                            " ".join(span1_text_list),
                            " ".join(span2_text_list),
                        ),
                        file=fout,
                    )

            print(file=fout)


def print_predictions_for_relation_decoding(outputs, file_path, vocab):
    with open(file_path, "w") as fout:
        for sent_output in outputs:
            seq_len = sent_output["seq_len"]
            assert "tokens" in sent_output
            tokens = [
                vocab.get_token_from_index(token, "tokens")
                for token in sent_output["tokens"][:seq_len]
            ]
            print("Token\t{}".format(" ".join(tokens)), file=fout)
            if "relation_label_matrix" in sent_output:
                for row in sent_output["relation_label_matrix"][:seq_len]:
                    print(
                        "Relation-Label-True\t{}".format(
                            " ".join(
                                [
                                    vocab.get_token_from_index(item, "ent_rel_id")
                                    for item in row[:seq_len]
                                ]
                            )
                        ),
                        file=fout,
                    )

            if "relation_label_preds" in sent_output:
                for row in sent_output["relation_label_preds"][:seq_len]:
                    print(
                        "Relation-Label-Pred\t{}".format(
                            " ".join(
                                [
                                    vocab.get_token_from_index(item, "ent_rel_id")
                                    for item in row[:seq_len]
                                ]
                            )
                        ),
                        file=fout,
                    )


def read_conjunctions(conjunctions_file: str):
    conj2sent = dict()
    file_path = conjunctions_file

    with open(file_path, "r") as fin:
        sent = 1
        currentSentText = ""
        for line in fin:
            if line == "\n":
                sent = 1
                continue
            elif sent == 1:
                currentSentText = line.replace("\n", "")
                sent = 0
            else:
                conj_sent = line.replace("\n", "")
                conj2sent[conj_sent] = currentSentText
    conj_sentences = list(conj2sent.keys())
    return conj_sentences, conj2sent


def print_extractions_jsonl_format(conjunctions_file: str, outputs, vocab, fout):
    conj_sentences, conj2sent = read_conjunctions(conjunctions_file)
    ext_texts = []
    for sent_output in outputs:
        extractions = {}
        seq_len = sent_output["seq_len"]
        assert "tokens" in sent_output
        tokens = [
            vocab.get_token_from_index(token, "tokens")
            for token in sent_output["tokens"][: seq_len - 6]
        ]
        sentence = " ".join(tokens)
        if sentence in conj_sentences:
            sentence = conj2sent[sentence]

        if "all_rel_preds" in sent_output:
            for (span1, span2), rel in sent_output["all_rel_preds"].items():
                if rel == "" or rel == " ":
                    continue
                if sent_output["all_ent_preds"][span1] == "Relation":
                    try:
                        if span2 in extractions[span1][rel]:
                            continue
                    except:  # noqa E722  # nosec B110
                        pass
                    try:
                        extractions[span1][rel].append(span2)
                    except:  # noqa E722  # nosec B110
                        extractions[span1] = defaultdict(list)
                        extractions[span1][rel].append(span2)
                else:
                    try:
                        if span1 in extractions[span2][rel]:
                            continue
                    except:  # noqa E722  # nosec B110
                        pass
                    try:
                        extractions[span2][rel].append(span1)
                    except:  # noqa E722  # nosec B110
                        extractions[span2] = defaultdict(list)
                        extractions[span2][rel].append(span1)
        to_remove_rel_spans = set()
        expand_rel = {}
        to_add = {}
        for rel_span1, d1 in extractions.items():
            for rel_span2, d2 in extractions.items():
                if rel_span1 != rel_span2 and not (
                    rel_span1 in to_remove_rel_spans or rel_span2 in to_remove_rel_spans
                ):
                    if d1["Subject"] == d2["Subject"] and d1["Object"] == d2["Object"]:
                        if rel_span1 in to_remove_rel_spans:
                            to_add[expand_rel[rel_span1] + rel_span2] = d1
                            to_remove_rel_spans.add(rel_span2)
                            to_remove_rel_spans.add(expand_rel[rel_span1])
                            expand_rel[rel_span2] = expand_rel[rel_span1] + rel_span2
                            expand_rel[rel_span1] = expand_rel[rel_span1] + rel_span2
                        elif rel_span2 in to_remove_rel_spans:
                            to_add[expand_rel[rel_span2] + rel_span1] = d1
                            to_remove_rel_spans.add(rel_span1)
                            to_remove_rel_spans.add(expand_rel[rel_span2])
                            expand_rel[rel_span1] = expand_rel[rel_span2] + rel_span1
                            expand_rel[rel_span2] = expand_rel[rel_span2] + rel_span1
                        else:
                            to_add[rel_span1 + rel_span2] = d1
                            expand_rel[rel_span1] = rel_span1 + rel_span2
                            expand_rel[rel_span2] = rel_span1 + rel_span2
                            to_remove_rel_spans.add(rel_span1)
                            to_remove_rel_spans.add(rel_span2)
        for tm in to_remove_rel_spans:
            del extractions[tm]
        for k, v in to_add.items():
            extractions[k] = v

        for rel_sp, d in extractions.items():
            if len(d["Subject"]) > 1:
                sorted_d_subject = sorted(d["Subject"], key=lambda x: x[0][0])
                sorted_d_subject = [x[0] for x in sorted_d_subject]
                subject_text = " ".join(
                    [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in sorted_d_subject
                    ]
                )
            elif len(d["Subject"]) == 1:
                subject_text = " ".join(
                    [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in d["Subject"][0]
                    ]
                )
            else:
                subject_text = ""
            if len(d["Object"]) > 1:
                sorted_d_object = sorted(d["Object"], key=lambda x: x[0][0])
                sorted_d_object = [x[0] for x in sorted_d_object]
                object_text = " ".join(
                    [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in sorted_d_object
                    ]
                )
            elif len(d["Object"]) == 1:
                object_text = " ".join(
                    [
                        " ".join(tokens[sub_span[0] : sub_span[1]])
                        for sub_span in d["Object"][0]
                    ]
                )
            else:
                object_text = ""
            rel_text = " ".join(
                [" ".join(tokens[sub_span[0] : sub_span[1]]) for sub_span in rel_sp]
            ).replace("[unused1]", "is")
            ext = json.dumps(
                {
                    "subject": subject_text,
                    "relation": rel_text,
                    "object": object_text,
                    "sentence": sentence,
                }
            )
            if ext not in ext_texts and (rel_text != "" and subject_text != ""):
                fout.write(ext)
                fout.write("\n")
            ext_texts.append(ext)
