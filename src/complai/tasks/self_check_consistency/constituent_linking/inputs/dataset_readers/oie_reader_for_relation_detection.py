import json
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


class ReaderForRelationDecoding:
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

                # if len(sentence['tokens']) != len(sentence['wordpiece_tokens_index']):
                #     logger.info(
                #         "sentence id: {} wordpiece_tokens_index length is not equal to tokens.".format(line['sentId']))
                #     # logger.info(sentence['tokens'], sentence['wordpiece_tokens_index'])
                #     # logger.info("lengths: {}, {}, {}".format(len(sentence['tokens']), len(sentence['wordpiece_tokens_index']),
                #     #              len(line['labelIds'])))
                #     continue

                if len(sentence["wordpiece_tokens"]) != len(
                    sentence["wordpiece_segment_ids"]
                ):
                    logger.info(
                        "sentence id: {} wordpiece_tokens length is not equal to wordpiece_segment_ids.".format(
                            line["sentId"]
                        )
                    )
                    continue

                state, results = self.get_label(line, len(sentence["tokens"]))
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

    def get_label(self, line, sentence_length):
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
        for entity in line["entityMentions"]:
            st, ed = entity["span_ids"][0], entity["span_ids"][-1]
            idx2ent[entity["emId"]] = ((st, ed), entity["text"])
            if (
                st > ed
                or st < 0
                or st > sentence_length
                or ed < 0
                or ed > sentence_length
            ):
                logger.info(
                    "sentence id: {} offset error. st: {}, ed: {}".format(
                        line["sentId"], st, ed
                    )
                )
                return False, results

            j = 0
            for i in range(st, ed):
                if entity_pos[i] != 0:
                    logger.info(
                        "sentence id: {} entity span overlap.".format(line["sentId"])
                    )
                    return False, results
                entity_pos[i] = 1
                j += 1

        if "relationMentions" not in line:
            logger.info(
                "sentence id: {} doesn't contain 'relationMentions'.".format(
                    line["sentId"]
                )
            )
            return False, results

        for relation in line["relationMentions"]:
            if (
                relation["arg1"]["emId"] not in idx2ent
                or relation["arg2"]["emId"] not in idx2ent
            ):
                logger.info(
                    "sentence id: {} entity not exists .".format(line["sentId"])
                )
                continue

        if "labelIds" not in line:
            logger.info(
                "sentence id: {} doesn't contain 'labelIds'.".format(line["sentId"])
            )
            return False, results

        if "relationIds" not in line:
            logger.info(
                "sentence id: {} doesn't contain 'relationIds'.".format(line["sentId"])
            )
            return False, results

        if "argumentIds" not in line:
            logger.info(
                "sentence id: {} doesn't contain 'argumentIds'.".format(line["sentId"])
            )
            return False, results

        results["label_ids"] = line["labelIds"]
        results["relation_ids"] = line["relationIds"]
        results["argument_ids"] = line["argumentIds"]

        return True, results

    def get_seq_lens(self):
        return self.seq_lens
