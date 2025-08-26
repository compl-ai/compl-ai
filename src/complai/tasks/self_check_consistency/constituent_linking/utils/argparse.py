import logging
import os

import configargparse

from complai.tasks.self_check_consistency.constituent_linking.utils.logging_utils import (
    init_logger,
)
from complai.tasks.self_check_consistency.constituent_linking.utils.parse_action import (
    StoreLoggingLevelAction,
)


class ConfigurationParer:
    """This class defines customized configuration parser"""

    def __init__(
        self,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        **kwargs,
    ):
        """This funtion decides config parser and formatter

        Keyword Arguments:
            config_file_parser_class {configargparse.ConfigFileParser} -- config file parser (default: {configargparse.YAMLConfigFileParser})
            formatter_class {configargparse.ArgumentDefaultsHelpFormatter} -- config formatter (default: {configargparse.ArgumentDefaultsHelpFormatter})
        """
        self.parser = configargparse.ArgumentParser(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs,
        )

    def add_save_cfgs(self):
        """This function adds saving path arguments: config file, model file..."""
        # config file configurations
        group = self.parser.add_argument_group("Config-File")
        group.add(
            "-config_file",
            "--config_file",
            required=False,
            is_config_file_arg=True,
            help="config file path",
        )

        # model file configurations
        group = self.parser.add_argument_group("Model-File")
        group.add(
            "-save_dir",
            "--save_dir",
            type=str,
            required=True,
            help="directory for saving checkpoints.",
        )

    def add_data_cfgs(self):
        """This function adds dataset arguments: data file path..."""
        self.parser.add(
            "-data_dir",
            "--data_dir",
            type=str,
            required=True,
            help="dataset directory.",
        )
        self.parser.add(
            "-train_file",
            "--train_file",
            type=str,
            required=False,
            help="train data file.",
        )
        self.parser.add(
            "-dev_file", "--dev_file", type=str, required=False, help="dev data file."
        )
        self.parser.add(
            "-test_file",
            "--test_file",
            type=str,
            required=False,
            help="test data file.",
        )
        self.parser.add(
            "-conjunctions_file",
            "--conjunctions_file",
            type=str,
            required=False,
            help="test conjunctions data file (produced by OpenIE6).",
        )
        self.parser.add(
            "-ent_rel_file",
            "--ent_rel_file",
            type=str,
            required=False,
            help="entity and relation file.",
        )
        self.parser.add(
            "-rel_file",
            "--rel_file",
            type=str,
            required=False,
            help="relation only file.",
        )
        self.parser.add(
            "-max_sent_len",
            "--max_sent_len",
            type=int,
            default=200,
            help="max sentence length.",
        )
        self.parser.add(
            "-max_wordpiece_len",
            "--max_wordpiece_len",
            type=int,
            default=512,
            help="max sentence length.",
        )
        self.parser.add("-test", "--test", action="store_true", help="testing mode")

    def add_model_cfgs(self):
        """This function adds model (network) arguments: embedding, hidden unit..."""
        # embedding configurations
        group = self.parser.add_argument_group("Embedding")
        group.add(
            "-embedding_model",
            "--embedding_model",
            type=str,
            choices=["bert", "pretrained"],
            default="bert",
            help="embedding model.",
        )
        group.add(
            "-bert_model_name",
            "--bert_model_name",
            type=str,
            required=False,
            help="bert model name.",
        )
        group.add(
            "-pretrained_model_name",
            "--pretrained_model_name",
            type=str,
            required=False,
            help="pretrained model name.",
        )
        group.add(
            "-bert_output_size",
            "--bert_output_size",
            type=int,
            default=768,
            help="bert output size.",
        )
        group.add(
            "-bert_dropout",
            "--bert_dropout",
            type=float,
            default=0.1,
            help="bert dropout rate.",
        )
        group.add(
            "--fine_tune",
            "--fine_tune",
            action="store_true",
            help="fine-tune pretrained model.",
        )

        # biaffine model
        group = self.parser.add_argument_group("Biaffine")
        group.add(
            "-max_span_length",
            "--max_span_length",
            type=int,
            default=10,
            help="maximum span length.",
        )
        group.add(
            "-mlp_hidden_size",
            "--mlp_hidden_size",
            type=int,
            default=768,
            help="mlp hidden units size.",
        )
        group.add(
            "-dropout", "--dropout", type=float, default=0.5, help="dropout rate."
        )
        group.add(
            "-separate_threshold",
            "--separate_threshold",
            type=float,
            default=1.07,
            help="the threshold for separating spans.",
        )
        group.add(
            "-logit_dropout",
            "--logit_dropout",
            type=float,
            default=0.1,
            help="logit dropout rate for robustness.",
        )

    def add_optimizer_cfgs(self):
        """This function adds optimizer arguments"""
        # gradient strategy
        self.parser.add(
            "-gradient_clipping",
            "--gradient_clipping",
            type=float,
            default=1.0,
            help="gradient clipping threshold.",
        )

        # learning rate
        self.parser.add(
            "--learning_rate",
            "-learning_rate",
            type=float,
            default=3e-5,
            help="Starting learning rate. "
            "Recommended settings: sgd = 1, adagrad = 0.1, "
            "adadelta = 1, adam = 0.001",
        )
        self.parser.add(
            "--bert_learning_rate",
            "-bert_learning_rate",
            type=float,
            default=3e-5,
            help="learning rate for bert, should be smaller than followed parts.",
        )
        self.parser.add(
            "-lr_decay_rate",
            "--lr_decay_rate",
            type=float,
            default=0.9,
            help="learn rate of layers decay rate.",
        )

        # Adam configurations
        group = self.parser.add_argument_group("Adam")
        group.add(
            "-adam_beta1",
            "--adam_beta1",
            type=float,
            default=0.9,
            help="The beta1 parameter used by Adam. "
            "Almost without exception a value of 0.9 is used in "
            "the literature, seemingly giving good results, "
            "so we would discourage changing this value from "
            "the default without due consideration.",
        )
        group.add(
            "-adam_beta2",
            "--adam_beta2",
            type=float,
            default=0.999,
            help="The beta2 parameter used by Adam. "
            "Typically a value of 0.999 is recommended, as this is "
            "the value suggested by the original paper describing "
            "Adam, and is also the value adopted in other frameworks "
            "such as Tensorflow and Kerras, i.e. see: "
            "https://www.tensorflow.org/api_docs/python/tf/train/Adam"
            "Optimizer or "
            "https://keras.io/optimizers/ . "
            'Whereas recently the paper "Attention is All You Need" '
            "suggested a value of 0.98 for beta2, this parameter may "
            "not work well for normal models / default "
            "baselines.",
        )
        group.add(
            "-adam_epsilon",
            "--adam_epsilon",
            type=float,
            default=1e-6,
            help="adam epsilon",
        )
        group.add(
            "-adam_weight_decay_rate",
            "--adam_weight_decay_rate",
            type=float,
            default=0.0,
            help="adam weight decay rate.",
        )
        group.add(
            "-adam_bert_weight_decay_rate",
            "--adam_bert_weight_decay_rate",
            type=float,
            default=0.0,
            help="adam weight decay rate of Bert module.",
        )

    def add_run_cfgs(self):
        """This function adds running arguments"""
        # training configurations
        group = self.parser.add_argument_group("Training")
        group.add("-seed", "--seed", type=int, default=5216, help="radom seed.")
        group.add(
            "-epochs", "--epochs", type=int, default=1000, help="training epochs."
        )
        group.add(
            "-pretrain_epochs",
            "--pretrain_epochs",
            type=int,
            default=0,
            help="pretrain epochs.",
        )
        group.add(
            "-warmup_rate",
            "--warmup_rate",
            type=float,
            default=0.0,
            help="warmup rate.",
        )
        group.add(
            "-early_stop",
            "--early_stop",
            type=int,
            default=50,
            help="early stop threshold.",
        )
        group.add(
            "-train_batch_size",
            "--train_batch_size",
            type=int,
            default=200,
            help="batch size during training.",
        )
        group.add(
            "-gradient_accumulation_steps",
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )

        # testing configurations
        group = self.parser.add_argument_group("Testing")
        group.add(
            "-test_batch_size",
            "--test_batch_size",
            type=int,
            default=100,
            help="batch size during testing.",
        )
        group.add(
            "-validate_every",
            "--validate_every",
            type=int,
            default=20000,
            help="output result every n samples during validating.",
        )

        # gpu configurations
        group = self.parser.add_argument_group("GPU")
        group.add(
            "-device",
            "--device",
            type=int,
            default=-1,
            help="cpu: device = -1, gpu: gpu device id(device >= 0).",
        )

        # logging configurations
        group = self.parser.add_argument_group("logging")
        group.add(
            "-root_log_level",
            "--root_log_level",
            type=str,
            action=StoreLoggingLevelAction,
            choices=StoreLoggingLevelAction.CHOICES,
            default="DEBUG",
            help="root logging out level.",
        )
        group.add(
            "-console_log_level",
            "--console_log_level",
            type=str,
            action=StoreLoggingLevelAction,
            choices=StoreLoggingLevelAction.CHOICES,
            default="NOTSET",
            help="console logging output level.",
        )
        group.add(
            "-log_file",
            "--log_file",
            type=str,
            required=True,
            help="logging file during running.",
        )
        group.add(
            "-file_log_level",
            "--file_log_level",
            type=str,
            action=StoreLoggingLevelAction,
            choices=StoreLoggingLevelAction.CHOICES,
            default="NOTSET",
            help="file logging output level.",
        )
        group.add(
            "-logging_steps",
            "--logging_steps",
            type=int,
            default=10,
            help="Logging every N update steps.",
        )

    def parse_args(self):
        """This function parses arguments and initializes logger

        Returns:
            dict -- config arguments
        """
        cfg = self.parser.parse_args()

        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)

        cfg.last_model_path = os.path.join(cfg.save_dir, "last_model")
        cfg.models = os.path.join(cfg.save_dir, "models")
        cfg.constituent_model_dir = os.path.join(cfg.models, "constituent")
        cfg.relation_model_dir = os.path.join(cfg.models, "relation")

        cfg.constituent_vocab = os.path.join(
            cfg.constituent_model_dir, "vocabulary.pickle"
        )
        cfg.relation_vocab = os.path.join(cfg.relation_model_dir, "vocabulary.pickle")
        cfg.constituent_model_path = os.path.join(cfg.constituent_model_dir, "ce_model")
        cfg.relation_model_path = os.path.join(cfg.relation_model_dir, "cl_model")

        if "carb" in cfg.test_file:
            cfg.separate_threshold = 1.25
            cfg.carb = True
        if "wire57" in cfg.test_file:
            cfg.separate_threshold = 1.05
            cfg.wire57 = True

        assert os.path.exists(cfg.data_dir), (
            f"dataset directory {cfg.data_dir} not exists !!!"
        )
        for file in ["train_file", "dev_file", "test_file"]:
            if getattr(cfg, file, None) is not None:
                setattr(cfg, file, os.path.join(cfg.data_dir, getattr(cfg, file, None)))

        if getattr(cfg, "log_file", None) is not None:
            cfg.log_file = os.path.join(cfg.save_dir, cfg.log_file)
            assert not os.path.exists(cfg.log_file), (
                f"log file {cfg.log_file} exists !!!"
            )

        init_logger(
            root_log_level=getattr(cfg, "root_log_level", logging.DEBUG),
            console_log_level=getattr(cfg, "console_log_level", logging.NOTSET),
            log_file=getattr(cfg, "log_file", None),
            log_file_level=getattr(cfg, "log_file_level", logging.NOTSET),
        )

        return cfg

    def format_values(self):
        return self.parser.format_values()
