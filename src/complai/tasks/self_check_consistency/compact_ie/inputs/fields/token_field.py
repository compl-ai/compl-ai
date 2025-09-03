import logging

from complai.tasks.self_check_consistency.compact_ie.inputs.fields.field import Field


logger = logging.getLogger(__name__)


class TokenField(Field):
    """Token field: regard sentence as token list"""

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
        """This function counts tokens in sentences,
        then update counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """
        if self.is_counting:
            for sentence in sentences:
                for token in sentence[self.source_key]:
                    counter[self.counter_namespace][str(token)] += 1

            logger.info(
                "Count sentences {} to update counter namespace {} successfully.".format(
                    self.source_key, self.counter_namespace
                )
            )

    def index(self, instance, vocab, sentences):
        """This function indexed token using vocabulary,
        then update instance

        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """
        for sentence in sentences:
            instance[self.namespace].append(
                [
                    vocab.get_token_index(token, self.vocab_namespace)
                    for token in sentence[self.source_key]
                ]
            )

        logger.info(
            "Index sentences {} to construct instance namespace {} successfully.".format(
                self.source_key, self.namespace
            )
        )
