import logging
import pickle

from bidict import bidict


logger = logging.getLogger(__name__)


class Vocabulary:
    """This class maps strings to integers, which also allow many namespaces"""

    DEFAULT_PAD_TOKEN = "*@PAD@*"
    DEFAULT_UNK_TOKEN = "*@UNK@*"

    def __init__(
        self,
        counters=dict(),
        min_count=dict(),
        pretrained_vocab=dict(),
        intersection_namespace=dict(),
        no_pad_namespace=list(),
        no_unk_namespace=list(),
        contain_pad_namespace=dict(),
        contain_unk_namespace=dict(),
    ):
        """Initialize vocabulary

        Keyword Arguments:
            counters {dict} -- multiple counter (default: {dict()})
            min_count {dict} -- min count dict (default: {dict()})
            pretrained_vocab {dict} -- pretrained vocabulary (default: {dict()})
            intersection_namespace {dict} -- intersection namespace correspond to pretrained vocabulary in case of too large pretrained vocabulary (default: {dict()})
            no_pad_namespace {list} -- no paddding namespace (default: {list()})
            no_unk_namespace {list} -- no unknown namespace (default: {list()})
            contain_pad_namespace {dict} -- contain padding token namespace (default: {dict()})
            contain_unk_namespace {dict} -- contain unknown token namespace (default: {dict()})
        """
        self.min_count = dict(min_count)
        self.intersection_namespace = dict(intersection_namespace)
        self.no_pad_namespace = set(no_pad_namespace)
        self.no_unk_namespace = set(no_unk_namespace)
        self.contain_pad_namespace = dict(contain_pad_namespace)
        self.contain_unk_namespace = dict(contain_unk_namespace)
        self.vocab = dict()

        self.extend_from_counter(
            counters, self.min_count, self.no_pad_namespace, self.no_unk_namespace
        )

        self.extend_from_pretrained_vocab(
            pretrained_vocab,
            self.intersection_namespace,
            self.no_pad_namespace,
            self.no_unk_namespace,
        )

        logger.info("Initialize vocabulary successfully.")

    def extend_from_pretrained_vocab(
        self,
        pretrained_vocab,
        intersection_namespace=dict(),
        no_pad_namespace=list(),
        no_unk_namespace=list(),
        contain_pad_namespace=dict(),
        contain_unk_namespace=dict(),
    ):
        """Extend vocabulary from pretrained vocab

        Arguments:
            pretrained_vocab {dict} -- pretrained vocabulary

        Keyword Arguments:
            intersection_namespace {dict} -- intersection namespace correspond to pretrained vocabulary in case of too large pretrained vocabulary (default: {dict()})
            no_pad_namespace {list} -- no paddding namespace (default: {list()})
            no_unk_namespace {list} -- no unknown namespace (default: {list()})
            contain_pad_namespace {dict} -- contain padding token namespace (default: {dict()})
            contain_unk_namespace {dict} -- contain unknown token namespace (default: {dict()})
        """
        self.intersection_namespace.update(dict(intersection_namespace))
        self.no_pad_namespace.update(set(no_pad_namespace))
        self.no_unk_namespace.update(set(no_unk_namespace))
        self.contain_pad_namespace.update(dict(contain_pad_namespace))
        self.contain_unk_namespace.update(dict(contain_unk_namespace))

        for namespace, vocab in pretrained_vocab.items():
            self.__namespace_init(namespace)
            is_intersection = namespace in self.intersection_namespace
            intersection_vocab = (
                self.vocab[self.intersection_namespace[namespace]]
                if is_intersection
                else []
            )
            for key, value in vocab.items():
                if not is_intersection or key in intersection_vocab:
                    self.vocab[namespace][key] = value

            logger.info(
                "Vocabulay {} (size: {}) was constructed successfully from pretrained_vocab.".format(
                    namespace, len(self.vocab[namespace])
                )
            )

    def extend_from_counter(
        self,
        counters,
        min_count=dict(),
        no_pad_namespace=list(),
        no_unk_namespace=list(),
        contain_pad_namespace=dict(),
        contain_unk_namespace=dict(),
    ):
        """Extend vocabulary from counter

        Arguments:
            counters {dict} -- multiply counter

        Keyword Arguments:
            min_count {dict} -- min count dict (default: {dict()})
            no_pad_namespace {list} -- no paddding namespace (default: {list()})
            no_unk_namespace {list} -- no unknown namespace (default: {list()})
            contain_pad_namespace {dict} -- contain padding token namespace (default: {dict()})
            contain_unk_namespace {dict} -- contain unknown token namespace (default: {dict()})
        """
        self.no_pad_namespace.update(set(no_pad_namespace))
        self.no_unk_namespace.update(set(no_unk_namespace))
        self.contain_pad_namespace.update(dict(contain_pad_namespace))
        self.contain_unk_namespace.update(dict(contain_unk_namespace))
        self.min_count.update(dict(min_count))

        for namespace, counter in counters.items():
            self.__namespace_init(namespace)
            for key in counter:
                minc = (
                    min_count[namespace] if min_count and namespace in min_count else 1
                )
                if counter[key] >= minc:
                    self.vocab[namespace][key] = len(self.vocab[namespace])

            logger.info(
                "Vocabulay {} (size: {}) was constructed successfully from counter.".format(
                    namespace, len(self.vocab[namespace])
                )
            )

    def add_tokens_to_namespace(self, tokens, namespace):
        """This function adds tokens to one namespace for extending vocabulary

        Arguments:
            tokens {list} -- token list
            namespace {str} -- namespace name
        """
        if namespace not in self.vocab:
            self.__namespace_init(namespace)
            logger.error("Add Namespace {} into vocabulary.".format(namespace))

        for token in tokens:
            if token not in self.vocab[namespace]:
                self.vocab[namespace][token] = len(self.vocab[namespace])

    def get_token_index(self, token, namespace):
        """This function gets token index in one namespace of vocabulary

        Arguments:
            token {str} -- token
            namespace {str} -- namespace name

        Raises:
            RuntimeError: namespace not exists

        Returns:
            int -- token index
        """
        if token in self.vocab[namespace]:
            return self.vocab[namespace][token]

        if namespace not in self.no_unk_namespace:
            return self.get_unknown_index(namespace)

        logger.error(
            "Can not find the index of {} from a no unknown token namespace {}.".format(
                token, namespace
            )
        )
        raise RuntimeError(
            "Can not find the index of {} from a no unknown token namespace {}.".format(
                token, namespace
            )
        )

    def get_token_from_index(self, index, namespace):
        """This function gets token using index in vocabulary

        Arguments:
            index {int} -- index
            namespace {str} -- namespace name

        Raises:
            RuntimeError: index out of range

        Returns:
            str -- token
        """
        if index < len(self.vocab[namespace]):
            return self.vocab[namespace].inv[index]

        logger.error(
            "The index {} is out of vocabulary {} range.".format(index, namespace)
        )
        raise RuntimeError(
            "The index {} is out of vocabulary {} range.".format(index, namespace)
        )

    def get_vocab_size(self, namespace):
        """This function gets the size of one namespace in vocabulary

        Arguments:
            namespace {str} -- namespace name

        Returns:
            int -- vocabulary size
        """
        return len(self.vocab[namespace])

    def get_all_namespaces(self):
        """This function gets all namespaces

        Returns:
            list -- all namespaces vocabulary contained
        """
        return set(self.vocab)

    def get_padding_index(self, namespace):
        """This function gets padding token index in one namespace of vocabulary

        Arguments:
            namespace {str} -- namespace name

        Raises:
            RuntimeError: no padding

        Returns:
            int -- padding index
        """
        if namespace not in self.vocab:
            raise RuntimeError("Namespace {} doesn't exist.".format(namespace))

        if namespace not in self.no_pad_namespace:
            if namespace not in self.contain_pad_namespace:
                return self.vocab[namespace][Vocabulary.DEFAULT_PAD_TOKEN]
            return self.vocab[namespace][self.contain_pad_namespace[namespace]]

        logger.error("Namespace {} doesn't has paddding token.".format(namespace))
        raise RuntimeError("Namespace {} doesn't has paddding token.".format(namespace))

    def get_unknown_index(self, namespace):
        """This function gets unknown token index in one namespace of vocabulary

        Arguments:
            namespace {str} -- namespace name

        Raises:
            RuntimeError: no unknown

        Returns:
            int -- unknown index
        """
        if namespace not in self.vocab:
            raise RuntimeError("Namespace {} doesn't exist.".format(namespace))

        if namespace not in self.no_unk_namespace:
            if namespace not in self.contain_unk_namespace:
                return self.vocab[namespace][Vocabulary.DEFAULT_UNK_TOKEN]
            return self.vocab[namespace][self.contain_unk_namespace[namespace]]

        logger.error("Namespace {} doesn't has unknown token.".format(namespace))
        raise RuntimeError("Namespace {} doesn't has unknown token.".format(namespace))

    def get_namespace_tokens(self, namesapce):
        """This function returns all tokens in one namespace

        Arguments:
            namesapce {str} -- namespce name

        Returns:
            dict_keys -- all tokens
        """
        return self.vocab[namesapce]

    def save(self, file_path):
        """This function saves vocabulary into file

        Arguments:
            file_path {str} -- file path
        """
        pickle.dump(self, open(file_path, "wb"))

    @classmethod
    def load(cls, file_path):
        """This function loads vocabulary from file

        Arguments:
            file_path {str} -- file path

        Returns:
            Vocabulary -- vocabulary
        """
        return pickle.load(open(file_path, "rb"), encoding="utf-8")

    def __namespace_init(self, namespace):
        """This function initializes a namespace,
        adds pad and unk token to one namespace of vacabulary

        Arguments:
            namespace {str} -- namespace
        """
        self.vocab[namespace] = bidict()

        if (
            namespace not in self.no_pad_namespace
            and namespace not in self.contain_pad_namespace
        ):
            self.vocab[namespace][Vocabulary.DEFAULT_PAD_TOKEN] = len(
                self.vocab[namespace]
            )

        if (
            namespace not in self.no_unk_namespace
            and namespace not in self.contain_unk_namespace
        ):
            self.vocab[namespace][Vocabulary.DEFAULT_UNK_TOKEN] = len(
                self.vocab[namespace]
            )
