import logging

from complai.tasks.self_check_consistency.constituent_linking.inputs.fields.field import (
    Field,
)


logger = logging.getLogger(__name__)


class MapTokenField(Field):
    """Map token field: preocess maping tokens"""

    def __init__(self, namespace, vocab_namespace, source_key, is_counting=True):
        """This function sets namesapce of field, vocab namespace for indexing, dataset source key

        Arguments:
            namespace {str} -- namesapce of field, counter namespace if constructing a counter
            vocab_namespace {str} -- vocab namespace for indexing
            source_key {str} -- indicate key in text data

        Keyword Arguments:
            is_counting {bool} -- decide constructing a counter or not (default: {True})
        """
        super().__init__()
        self.namespace = str(namespace)
        self.counter_namespace = str(namespace)
        self.vocab_namespace = str(vocab_namespace)
        self.source_key = str(source_key)
        self.is_counting = is_counting

    def count_vocab_items(self, counter, sentences):
        """This function counts dict's values in sentences,
        then update counter, each sentence is a dict

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing, list of dict
        """
        if self.is_counting:
            for sentence in sentences:
                for value in sentence[self.source_key].values():
                    counter[self.counter_namespace][str(value)] += 1

            logger.info(
                "Count sentences {} to update counter namespace {} successfully.".format(
                    self.source_key, self.counter_namespace
                )
            )

    def index(self, instance, vocab, sentences):
        """This function indexes token using vocabulary, then update instance

        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """
        for sentence in sentences:
            instance[self.namespace].append(
                {
                    key: vocab.get_token_index(value, self.vocab_namespace)
                    for key, value in sentence[self.source_key].items()
                }
            )

        logger.info(
            "Index sentences {} to construct instance namespace {} successfully.".format(
                self.source_key, self.namespace
            )
        )
