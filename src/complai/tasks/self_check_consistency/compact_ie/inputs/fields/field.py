from abc import ABC
from abc import abstractclassmethod


class Field(ABC):
    """Abstract class `Field` define one indexing method,
    genenrate counter from raw text data and index token in raw text data

    Arguments:
        ABC {ABC} -- abstract base class
    """

    @abstractclassmethod
    def count_vocab_items(self, counter, sentences):
        """This function constructs counter using each sentence content,
        prepare for vocabulary

        Arguments:
            counter {dict} -- element count dict
            sentences {list} -- text data
        """
        raise NotImplementedError

    @abstractclassmethod
    def index(self, instance, voacb, sentences):
        """This function constrcuts instance using sentences and vocabulary,
        each namespace is a mappping method using different type data

        Arguments:
            instance {dict} -- collections of various fields
            voacb {dict} -- vocabulary
            sentences {list} -- text data
        """
        raise NotImplementedError
