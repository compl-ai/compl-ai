import logging

from complai.tasks.self_check_consistency.compact_ie.inputs.fields.field import Field


logger = logging.getLogger(__name__)


class RawTokenField(Field):
    """This Class preserves raw text of tokens"""

    def __init__(self, namespace, source_key):
        """This function sets namesapce of field, dataset source key

        Arguments:
            namespace {str} -- namesapce of field
            source_key {str} -- indicate key in text data
        """
        super().__init__()
        self.namespace = str(namespace)
        self.source_key = str(source_key)

    def count_vocab_items(self, counter, sentences):
        """`RawTokenField` doesn't update counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """
        pass

    def index(self, instance, vocab, sentences):
        """This function doesn't use vocabulary,
        perserve raw text of sentences(tokens)

        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """
        for sentence in sentences:
            instance[self.namespace].append(
                [token for token in sentence[self.source_key]]
            )

        logger.info(
            "Index sentences {} to construct instance namespace {} successfully.".format(
                self.source_key, self.namespace
            )
        )
