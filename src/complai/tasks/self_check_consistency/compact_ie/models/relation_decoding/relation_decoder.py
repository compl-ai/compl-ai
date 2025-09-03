import logging

import torch
import torch.nn as nn

from complai.tasks.self_check_consistency.compact_ie.models.embedding_models.bert_embedding_model import (
    BertEmbedModel,
)

# from complai.tasks.self_check_consistency.compact_ie.models.embedding_models.pretrained_embedding_model import (
#     PretrainedEmbedModel,
# )
# from complai.tasks.self_check_consistency.compact_ie.modules.token_embedders.bert_encoder import (
#     BertLinear,
# )
from complai.tasks.self_check_consistency.compact_ie.modules.token_embedders.bert_encoder import (
    BertLayerNorm,
)


# from transformers import BertModel

logger = logging.getLogger(__name__)


class RelDecoder(nn.Module):
    def __init__(
        self,
        vocab,
        ent_rel_file,
        max_span_length,
        device,
        logit_dropout,
        bert_model_name,
        fine_tune,
        bert_output_size,
        bert_dropout,
    ):
        """__init__ constructs `EntRelJointDecoder` components and
        sets `EntRelJointDecoder` parameters. This class adopts a joint
        decoding algorithm for entity relation joint decoing and facilitates
        the interaction between entity and relation.
        """
        super().__init__()

        self.num_labels = 3
        self.vocab = vocab
        self.max_span_length = max_span_length
        self.device = device

        # if cfg.rel_embedding_model == 'bert':
        self.embedding_model = BertEmbedModel(
            bert_model_name, fine_tune, bert_output_size, bert_dropout, vocab, True
        )
        # elif cfg.rel_embedding_model == 'pretrained':
        #     self.embedding_model = PretrainedEmbedModel(cfg, vocab)
        self.encoder_output_size = self.embedding_model.get_hidden_size()

        self.layer_norm = BertLayerNorm(self.encoder_output_size * 2)
        self.classifier = nn.Linear(self.encoder_output_size * 2, self.num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

        if logit_dropout > 0:
            self.dropout = nn.Dropout(p=logit_dropout)
        else:
            self.dropout = lambda x: x

        self.none_idx = self.vocab.get_token_index("None", "ent_rel_id")

        self.rel_label = torch.LongTensor(ent_rel_file["relation"])
        if self.device > -1:
            self.rel_label = self.rel_label.cuda(device=self.device, non_blocking=True)

        self.element_loss = nn.CrossEntropyLoss()

    def forward(self, batch_inputs):
        """Forward

        Arguments:
            batch_inputs {dict} -- batch input data

        Returns:
            dict -- results: ent_loss, ent_pred
        """
        results = {}

        self.embedding_model(batch_inputs)
        batch_seq_tokens_encoder_repr = batch_inputs["seq_encoder_reprs"]
        relation_tokens = batch_seq_tokens_encoder_repr[
            torch.arange(batch_seq_tokens_encoder_repr.shape[0]).unsqueeze(-1),
            batch_inputs["relation_ids"],
        ]
        argument_tokens = batch_seq_tokens_encoder_repr[
            torch.arange(batch_seq_tokens_encoder_repr.shape[0]).unsqueeze(-1),
            batch_inputs["argument_ids"],
        ]
        batch_input_rep = torch.cat((relation_tokens, argument_tokens), dim=-1)
        batch_input_rep = self.layer_norm(batch_input_rep)
        batch_input_rep = self.dropout(batch_input_rep)
        batch_logits = self.classifier(batch_input_rep)

        if not self.training:
            results["label_preds"] = (
                torch.argmax(batch_logits, dim=-1) * batch_inputs["label_ids_mask"]
            )
            return results

        results["loss"] = self.element_loss(
            batch_logits[batch_inputs["label_ids_mask"]],
            batch_inputs["label_ids"][batch_inputs["label_ids_mask"]],
        )

        return results
