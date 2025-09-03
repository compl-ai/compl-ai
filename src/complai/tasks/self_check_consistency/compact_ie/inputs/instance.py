import logging


logger = logging.getLogger(__name__)


class Instance:
    """`Instance` is the collection of multiple `Field`"""

    def __init__(self, fields):
        """This function initializes instance

        Arguments:
            fields {list} -- field list
        """
        self.fields = list(fields)
        self.instance = {}
        for field in self.fields:
            self.instance[field.namespace] = []
        self.vocab_dict = {}
        self.vocab_index()

    def __getitem__(self, namespace):
        if namespace not in self.instance:
            logger.error("can not find the namespace {} in instance.".format(namespace))
            raise RuntimeError(
                "can not find the namespace {} in instance.".format(namespace)
            )
        else:
            self.instance.get(namespace, None)

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def add_fields(self, fields):
        """This function adds fields to instance

        Arguments:
            field {Field} -- field list
        """
        for field in fields:
            if field.namesapce not in self.instance:
                self.fields.append(field)
                self.instance[field.namesapce] = []
            else:
                logger.warning("Field {} has been added before.".format(field.name))

        self.vocab_index()

    def count_vocab_items(self, counter, sentences):
        """This funtion constructs multiple namespace in counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """
        for field in self.fields:
            field.count_vocab_items(counter, sentences)

    def index(self, vocab, sentences):
        """This funtion indexes token using vocabulary,
        then update instance

        Arguments:
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """
        for field in self.fields:
            field.index(self.instance, vocab, sentences)

    def get_instance(self):
        """This function get instance

        Returns:
            dict -- instance
        """
        return self.instance

    def get_size(self):
        """This funtion gets the size of instance

        Returns:
            int -- instance size
        """
        return len(self.instance[self.fields[0].namespace])

    def vocab_index(self):
        """This function constructs vocabulary dict of fields"""
        for field in self.fields:
            if hasattr(field, "vocab_namespace"):
                self.vocab_dict[field.namespace] = field.vocab_namespace

    def get_vocab_dict(self):
        """This function gets the vocab dict of instance

        Returns:
            dict -- vocab dict
        """
        return self.vocab_dict
