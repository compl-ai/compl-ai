import json
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


def split_span(span):
    sub_spans = [[span[0]]]
    for i in range(1, len(span)):
        if span[i - 1] == span[i] - 1:
            sub_spans[-1].append(span[i])
        else:
            sub_spans.append([span[i]])
    return sub_spans


class OIE4ReaderForJointDecoding:
    """Define text data reader and preprocess data for entity relation
    joint decoding on ACE dataset.
    """

    def __init__(self, file_path, is_test=False, max_len=dict()):
        """This function defines file path and some settings

        Arguments:
            file_path {str} -- file path

        Keyword Arguments:
            is_test {bool} -- indicate training or testing (default: {False})
            max_len {dict} -- max length for some namespace (default: {dict()})
        """
        self.file_path = file_path
        self.is_test = is_test
        self.max_len = dict(max_len)
        self.seq_lens = defaultdict(list)

    def __iter__(self):
        """Generator function"""
        with open(self.file_path, "r") as fin:
            for line in fin:
                line = json.loads(line)
                sentence = {}

                state, results = self.get_tokens(line)
                self.seq_lens["tokens"].append(len(results["tokens"]))
                if not state or (
                    "tokens" in self.max_len
                    and len(results["tokens"]) > self.max_len["tokens"]
                    and not self.is_test
                ):
                    if not self.is_test:
                        continue
                sentence.update(results)

                state, results = self.get_wordpiece_tokens(line)
                self.seq_lens["wordpiece_tokens"].append(
                    len(results["wordpiece_tokens"])
                )
                if not state or (
                    "wordpiece_tokens" in self.max_len
                    and len(results["wordpiece_tokens"])
                    > self.max_len["wordpiece_tokens"]
                ):
                    if not self.is_test:
                        continue
                sentence.update(results)

                if len(sentence["tokens"]) != len(sentence["wordpiece_tokens_index"]):
                    logger.info(
                        "sentence id: {} wordpiece_tokens_index length is not equal to tokens.".format(
                            line["sentId"]
                        )
                    )
                    continue

                if len(sentence["wordpiece_tokens"]) != len(
                    sentence["wordpiece_segment_ids"]
                ):
                    logger.info(
                        "sentence id: {} wordpiece_tokens length is not equal to wordpiece_segment_ids.".format(
                            line["sentId"]
                        )
                    )
                    continue

                state, results = self.get_entity_relation_label(
                    line, len(sentence["tokens"])
                )
                for key, result in results.items():
                    self.seq_lens[key].append(len(result))
                    if key in self.max_len and len(result) > self.max_len[key]:
                        state = False
                if not state:
                    continue
                sentence.update(results)

                yield sentence

    def get_tokens(self, line):
        """This function splits text into tokens

        Arguments:
            line {dict} -- text

        Returns:
            bool -- execute state
            dict -- results: tokens
        """
        results = {}

        if "sentText" not in line:
            logger.info(
                "sentence id: {} doesn't contain 'sentText'.".format(line["sentId"])
            )
            return False, results

        results["text"] = line["sentText"]

        if "tokens" in line:
            results["tokens"] = line["tokens"]
        else:
            results["tokens"] = line["sentText"].strip().split(" ")

        return True, results

    def get_wordpiece_tokens(self, line):
        """This function splits wordpiece text into wordpiece tokens

        Arguments:
            line {dict} -- text

        Returns:
            bool -- execute state
            dict -- results: tokens
        """
        results = {}

        if (
            "wordpieceSentText" not in line
            or "wordpieceTokensIndex" not in line
            or "wordpieceSegmentIds" not in line
        ):
            logger.info(
                "sentence id: {} doesn't contain 'wordpieceSentText' or 'wordpieceTokensIndex' or 'wordpieceSegmentIds'.".format(
                    line["sentId"]
                )
            )
            return False, results

        wordpiece_tokens = line["wordpieceSentText"].strip().split(" ")
        results["wordpiece_tokens"] = wordpiece_tokens
        results["wordpiece_tokens_index"] = [
            span[0] for span in line["wordpieceTokensIndex"]
        ]
        results["wordpiece_segment_ids"] = list(line["wordpieceSegmentIds"])

        return True, results

    def get_entity_relation_label(self, line, sentence_length):
        """This function constructs mapping relation from span to entity label
        and span pair to relation label, and joint entity relation label matrix.

        Arguments:
            line {dict} -- text
            sentence_length {int} -- sentence length

        Returns:
            bool -- execute state
            dict -- ent2rel: entity span mapping to entity label,
            span2rel: two entity span mapping to relation label,
            joint_label_matrix: joint entity relation label matrix
        """
        results = {}

        if "entityMentions" not in line:
            logger.info(
                "sentence id: {} doesn't contain 'entityMentions'.".format(
                    line["sentId"]
                )
            )
            return False, results

        entity_pos = [0] * sentence_length
        idx2ent = {}
        span2ent = {}

        separate_positions = []
        for entity in line["entityMentions"]:
            entity_sub_spans = []
            # span = entity['span_ids']
            st, ed = entity["span_ids"][0], entity["span_ids"][-1]
            sub_spans = split_span(entity["span_ids"])
            if len(sub_spans) == 1:
                if st > 0:
                    separate_positions.append(st - 1)
                if ed < sentence_length - 1:
                    separate_positions.append(ed)
                entity_sub_spans.append((st, ed + 1))
            else:
                # noncontinuous spans
                for sub in sub_spans:
                    if sub[0] > 0:
                        separate_positions.append(sub[0] - 1)
                    if sub[-1] < sentence_length - 1:
                        separate_positions.append(sub[-1])
                    entity_sub_spans.append((sub[0], sub[-1] + 1))

            idx2ent[entity["emId"]] = (tuple(entity_sub_spans), entity["text"])
            # if len(sub) > 1:
            #     separate_positions.append(sub[-1])
            # add start and end of an span for determining split positions
            # (in our case, should add more separate positions due to non-continues spans)
            # st, ed = entity['offset']
            # if st != 0:
            #     separate_positions.append(st - 1)
            # if ed != sentence_length:
            #     separate_positions.append(ed - 1)

            # idx2ent[entity['emId']] = ((st, ed), entity['text'])
            # if st >= ed + 1 or st < 0 or st > sentence_length or ed < 0 or ed > sentence_length:
            #     logger.info("sentence id: {} offset error: {}'.".format(line['sentId'], entity['span_ids']))
            # return False, results

            span2ent[tuple(entity_sub_spans)] = entity["label"]

            j = 0
            for s_i in entity["span_ids"]:
                if entity_pos[s_i] != 0:
                    logger.info(
                        "sentence id: {} entity span overlap. {}".format(
                            line["sentId"], entity["span_ids"]
                        )
                    )
                    return False, results
                entity_pos[s_i] = 1
                j += 1

        separate_positions = list(set(separate_positions))
        results["separate_positions"] = sorted(separate_positions)
        results["span2ent"] = span2ent

        if "relationMentions" not in line:
            logger.info(
                "sentence id: {} doesn't contain 'relationMentions'.".format(
                    line["sentId"]
                )
            )
            return False, results

        span2rel = {}
        for relation in line["relationMentions"]:
            if (
                relation["arg1"]["emId"] not in idx2ent
                or relation["arg2"]["emId"] not in idx2ent
            ):
                logger.info(
                    "sentence id: {} entity not exists .".format(line["sentId"])
                )
                continue

            entity1_span, entity1_text = idx2ent[relation["arg1"]["emId"]]
            entity2_span, entity2_text = idx2ent[relation["arg2"]["emId"]]

            if (
                entity1_text != relation["arg1"]["text"]
                or entity2_text != relation["arg2"]["text"]
            ):
                logger.info(
                    "sentence id: {} entity text doesn't match realtiaon text.".format(
                        line["sentId"]
                    )
                )
                return False, None

            span2rel[(entity1_span, entity2_span)] = relation["label"]

        results["span2rel"] = span2rel

        if "jointLabelMatrix" not in line:
            logger.info(
                "sentence id: {} doesn't contain 'jointLabelMatrix'.".format(
                    line["sentId"]
                )
            )
            return False, results

        results["joint_label_matrix"] = line["jointLabelMatrix"]
        return True, results

    def get_seq_lens(self):
        return self.seq_lens
