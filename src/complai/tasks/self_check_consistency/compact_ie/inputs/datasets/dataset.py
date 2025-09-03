import copy
import logging
import random


logger = logging.getLogger(__name__)


class Dataset:
    """This class constructs dataset for multiple date file"""

    def __init__(self, name, instance_dict=dict()):
        """This function initializes a dataset,
        define dataset name, this dataset contains multiple readers, as datafiles.

        Arguments:
            name {str} -- dataset name

        Keyword Arguments:
            instance_dict {dict} -- instance settings (default: {dict()})
        """
        self.dataset_name = name
        self.datasets = dict()
        self.instance_dict = copy.deepcopy(instance_dict)

    def add_instance(self, name, instance, reader, is_count=False, is_train=False):
        """This function adds a instance to dataset

        Arguments:
            name {str} -- intance name
            instance {Instance} -- instance
            reader {DatasetReader} -- reader correspond to instance

        Keyword Arguments:
            is_count {bool} -- instance paticipates in counting or not (default: {False})
            is_train {bool} -- instance is training data or not (default: {False})
        """
        self.instance_dict[name] = {
            "instance": instance,
            "reader": reader,
            "is_count": is_count,
            "is_train": is_train,
        }

    def build_dataset(
        self,
        vocab,
        counter=None,
        min_count=dict(),
        pretrained_vocab=None,
        intersection_namespace=dict(),
        no_pad_namespace=list(),
        no_unk_namespace=list(),
        contain_pad_namespace=dict(),
        contain_unk_namespace=dict(),
        tokens_to_add=None,
    ):
        """This function bulids dataset

        Arguments:
            vocab {Vocabulary} -- vocabulary

        Keyword Arguments:
            counter {dict} -- counter (default: {None})
            min_count {dict} -- min count for each namespace (default: {dict()})
            pretrained_vocab {dict} -- pretrained vocabulary (default: {None})
            intersection_namespace {dict} -- intersection vocabulary namespace correspond to
            pretrained vocabulary in case of too large pretrained vocabulary (default: {dict()})
            no_pad_namespace {list} -- no padding vocabulary namespace (default: {list()})
            no_unk_namespace {list} -- no unknown vocabulary namespace (default: {list()})
            contain_pad_namespace {dict} -- contain padding token vocabulary namespace (default: {dict()})
            contain_unk_namespace {dict} -- contain unknown token vocabulary namespace (default: {dict()})
            tokens_to_add {dict} -- tokens need to be added to vocabulary (default: {None})
        """
        # construct counter
        if counter is not None:
            for instance_name, instance_settting in self.instance_dict.items():
                if instance_settting["is_count"]:
                    instance_settting["instance"].count_vocab_items(
                        counter, instance_settting["reader"]
                    )

            # construct vocabulary from counter
            vocab.extend_from_counter(
                counter,
                min_count,
                no_pad_namespace,
                no_unk_namespace,
                contain_pad_namespace,
                contain_unk_namespace,
            )

        # add extra tokens, this operation should be executeed before adding pretrained_vocab
        if tokens_to_add is not None:
            for namespace, tokens in tokens_to_add.items():
                vocab.add_tokens_to_namespace(tokens, namespace)

        # construct vocabulary from pretained vocabulary
        if pretrained_vocab is not None:
            vocab.extend_from_pretrained_vocab(
                pretrained_vocab,
                intersection_namespace,
                no_pad_namespace,
                no_unk_namespace,
                contain_pad_namespace,
                contain_unk_namespace,
            )

        self.vocab = vocab

        for instance_name, instance_settting in self.instance_dict.items():
            instance_settting["instance"].index(self.vocab, instance_settting["reader"])
            self.datasets[instance_name] = instance_settting["instance"].get_instance()
            self.instance_dict[instance_name]["size"] = instance_settting[
                "instance"
            ].get_size()
            self.instance_dict[instance_name]["vocab_dict"] = instance_settting[
                "instance"
            ].get_vocab_dict()

            logger.info(
                "{} dataset size: {}.".format(
                    instance_name, self.instance_dict[instance_name]["size"]
                )
            )
            for key, seq_len in instance_settting["reader"].get_seq_lens().items():
                logger.info(
                    "{} dataset's {}: max_len={}, min_len={}.".format(
                        instance_name, key, max(seq_len), min(seq_len)
                    )
                )

    def get_batch(self, instance_name, batch_size, sort_namespace=None):
        """get_batch gets batch data and padding

        Arguments:
            instance_name {str} -- instance name
            batch_size {int} -- batch size

        Keyword Arguments:
            sort_namespace {str} -- sort samples key, meanwhile calculate sequence length if not None, while keep None means that no sorting (default: {None})

        Yields:
            int -- epoch
            dict -- batch data
        """
        if instance_name not in self.instance_dict:
            logger.error(
                "can not find instance name {} in datasets.".format(instance_name)
            )
            return

        dataset = self.datasets[instance_name]

        if sort_namespace is not None and sort_namespace not in dataset:
            logger.error(
                "can not find sort namespace {} in datasets instance {}.".format(
                    sort_namespace, instance_name
                )
            )

        size = self.instance_dict[instance_name]["size"]
        vocab_dict = self.instance_dict[instance_name]["vocab_dict"]
        ids = list(range(size))
        if self.instance_dict[instance_name]["is_train"]:
            random.shuffle(ids)
        epoch = 1
        cur = 0

        while True:
            if cur >= size:
                epoch += 1
                if not self.instance_dict[instance_name]["is_train"] and epoch > 1:
                    break
                random.shuffle(ids)
                cur = 0

            sample_ids = ids[cur : cur + batch_size]
            cur += batch_size

            if sort_namespace is not None:
                sample_ids = [
                    (idx, len(dataset[sort_namespace][idx])) for idx in sample_ids
                ]
                sample_ids = sorted(sample_ids, key=lambda x: x[1], reverse=True)
                sorted_ids = [idx for idx, _ in sample_ids]
            else:
                sorted_ids = sample_ids

            batch = {}

            for namespace in dataset:
                batch[namespace] = []

                if namespace in self.wo_padding_namespace:
                    for id in sorted_ids:
                        batch[namespace].append(dataset[namespace][id])
                else:
                    if namespace in vocab_dict:
                        padding_idx = self.vocab.get_padding_index(
                            vocab_dict[namespace]
                        )
                    else:
                        padding_idx = 0

                    batch_namespace_len = [
                        len(dataset[namespace][id]) for id in sorted_ids
                    ]
                    max_namespace_len = max(batch_namespace_len)
                    batch[namespace + "_lens"] = batch_namespace_len
                    batch[namespace + "_mask"] = []

                    if isinstance(dataset[namespace][0][0], list):
                        max_char_len = 0
                        for id in sorted_ids:
                            max_char_len = max(
                                max_char_len,
                                max(len(item) for item in dataset[namespace][id]),
                            )
                        for id in sorted_ids:
                            padding_sent = []
                            mask = []
                            for item in dataset[namespace][id]:
                                padding_sent.append(
                                    item + [padding_idx] * (max_char_len - len(item))
                                )
                                mask.append(
                                    [1] * len(item) + [0] * (max_char_len - len(item))
                                )
                            padding_sent = padding_sent + [
                                [padding_idx] * max_char_len
                            ] * (max_namespace_len - len(dataset[namespace][id]))
                            mask = mask + [[0] * max_char_len] * (
                                max_namespace_len - len(dataset[namespace][id])
                            )
                            batch[namespace].append(padding_sent)
                            batch[namespace + "_mask"].append(mask)
                    else:
                        for id in sorted_ids:
                            batch[namespace].append(
                                dataset[namespace][id]
                                + [padding_idx]
                                * (max_namespace_len - len(dataset[namespace][id]))
                            )
                            batch[namespace + "_mask"].append(
                                [1] * len(dataset[namespace][id])
                                + [0]
                                * (max_namespace_len - len(dataset[namespace][id]))
                            )

            yield epoch, batch

    def get_dataset_size(self, instance_name):
        """This function gets dataset size

        Arguments:
            instance_name {str} -- instance name

        Returns:
            int -- dataset size
        """
        return self.instance_dict[instance_name]["size"]

    def set_wo_padding_namespace(self, wo_padding_namespace):
        """set_wo_padding_namespace sets without paddding namespace

        Args:
            wo_padding_namespace (list): without padding namespace
        """
        self.wo_padding_namespace = wo_padding_namespace
