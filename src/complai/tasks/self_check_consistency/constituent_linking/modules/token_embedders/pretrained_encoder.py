import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from complai.tasks.self_check_consistency.constituent_linking.modules.token_embedders.bert_encoder import (
    BertLinear,
)
from complai.tasks.self_check_consistency.constituent_linking.utils.nn_utils import gelu


logger = logging.getLogger(__name__)


class PretrainedEncoder(nn.Module):
    """This class using pre-trained model to encode token,
    then fine-tuning the pre-trained model
    """

    def __init__(
        self,
        pretrained_model_name,
        trainable=False,
        output_size=0,
        activation=gelu,
        dropout=0.0,
    ):
        """This function initialize pertrained model

        Arguments:
            pretrained_model_name {str} -- pre-trained model name

        Keyword Arguments:
            output_size {float} -- output size (default: {None})
            activation {nn.Module} -- activation function (default: {gelu})
            dropout {float} -- dropout rate (default: {0.0})
        """
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        logger.info(
            "Load pre-trained model {} successfully.".format(pretrained_model_name)
        )

        self.output_size = output_size

        if trainable:
            logger.info(
                "Start fine-tuning pre-trained model {}.".format(pretrained_model_name)
            )
        else:
            logger.info(
                "Keep fixed pre-trained model {}.".format(pretrained_model_name)
            )

        for param in self.pretrained_model.parameters():
            param.requires_grad = trainable

        if self.output_size > 0:
            self.mlp = BertLinear(
                input_size=self.pretrained_model.config.hidden_size,
                output_size=self.output_size,
                activation=activation,
            )
        else:
            self.output_size = self.pretrained_model.config.hidden_size
            self.mlp = lambda x: x

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    def get_output_dims(self):
        return self.output_size

    def forward(self, seq_inputs, token_type_inputs=None):
        """Forward calculates forward propagation results, get token embedding

        Args:
            seq_inputs {tensor} -- sequence inputs (tokenized)
            token_type_inputs (tensor, optional): token type inputs. Defaults to None.

        Returns:
            tensor: bert output for tokens
        """
        if token_type_inputs is None:
            token_type_inputs = torch.zeros_like(seq_inputs)
        mask_inputs = (seq_inputs != 0).long()

        outputs = self.pretrained_model(
            input_ids=seq_inputs,
            token_type_ids=token_type_inputs,
            attention_mask=mask_inputs,
        )
        last_hidden_state = outputs[0]
        pooled_output = outputs[1]

        return self.dropout(self.mlp(last_hidden_state)), self.dropout(
            self.mlp(pooled_output)
        )
