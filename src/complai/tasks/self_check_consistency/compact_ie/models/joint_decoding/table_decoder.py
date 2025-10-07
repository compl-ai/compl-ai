import copy
import logging

import numpy as np
import torch
import torch.nn as nn

from complai.tasks.self_check_consistency.compact_ie.models.embedding_models.bert_embedding_model import (
    BertEmbedModel,
)
from complai.tasks.self_check_consistency.compact_ie.models.embedding_models.pretrained_embedding_model import (
    PretrainedEmbedModel,
)
from complai.tasks.self_check_consistency.compact_ie.modules.token_embedders.bert_encoder import (
    BertLinear,
)


logger = logging.getLogger(__name__)


class EntRelJointDecoder(nn.Module):
    def __init__(self, cfg, vocab, ent_rel_file):
        """__init__ constructs `EntRelJointDecoder` components and
        sets `EntRelJointDecoder` parameters. This class adopts a joint
        decoding algorithm for entity relation joint decoding and facilitates
        the interaction between entity and relation.

        Args:
            cfg (dict): config parameters for constructing multiple models
            vocab (Vocabulary): vocabulary
            ent_rel_file (dict): entity and relation file (joint id, entity id, relation id, symmetric id, asymmetric id)
        """
        super().__init__()
        self.vocab = vocab
        self.max_span_length = cfg.max_span_length
        self.activation = nn.GELU()
        self.device = cfg.device
        self.separate_threshold = cfg.separate_threshold

        if cfg.embedding_model == "bert":
            self.embedding_model = BertEmbedModel(cfg, vocab)
        elif cfg.embedding_model == "pretrained":
            self.embedding_model = PretrainedEmbedModel(cfg, vocab)
        self.encoder_output_size = self.embedding_model.get_hidden_size()

        self.head_mlp = BertLinear(
            input_size=self.encoder_output_size,
            output_size=cfg.mlp_hidden_size,
            activation=self.activation,
            dropout=cfg.dropout,
        )
        self.tail_mlp = BertLinear(
            input_size=self.encoder_output_size,
            output_size=cfg.mlp_hidden_size,
            activation=self.activation,
            dropout=cfg.dropout,
        )

        self.U = nn.Parameter(
            torch.FloatTensor(
                self.vocab.get_vocab_size("ent_rel_id"),
                cfg.mlp_hidden_size + 1,
                cfg.mlp_hidden_size + 1,
            )
        )
        self.U.data.zero_()

        if cfg.logit_dropout > 0:
            self.logit_dropout = nn.Dropout(p=cfg.logit_dropout)
        else:
            self.logit_dropout = lambda x: x

        self.none_idx = self.vocab.get_token_index("None", "ent_rel_id")

        self.symmetric_label = torch.LongTensor(ent_rel_file["symmetric"])
        self.asymmetric_label = torch.LongTensor(ent_rel_file["asymmetric"])
        self.ent_label = torch.LongTensor(ent_rel_file["entity"])
        self.rel_label = torch.LongTensor(ent_rel_file["relation"])
        if self.device > -1:
            self.symmetric_label = self.symmetric_label.cuda(
                device=self.device, non_blocking=True
            )
            self.asymmetric_label = self.asymmetric_label.cuda(
                device=self.device, non_blocking=True
            )
            self.ent_label = self.ent_label.cuda(device=self.device, non_blocking=True)
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
        batch_seq_tokens_lens = batch_inputs["tokens_lens"]
        batch_tokens = batch_inputs["tokens"]

        self.embedding_model(batch_inputs)
        batch_seq_tokens_encoder_repr = batch_inputs["seq_encoder_reprs"]

        batch_seq_tokens_head_repr = self.head_mlp(batch_seq_tokens_encoder_repr)
        batch_seq_tokens_head_repr = torch.cat(
            [
                batch_seq_tokens_head_repr,
                torch.ones_like(batch_seq_tokens_head_repr[..., :1]),
            ],
            dim=-1,
        )
        batch_seq_tokens_tail_repr = self.tail_mlp(batch_seq_tokens_encoder_repr)
        batch_seq_tokens_tail_repr = torch.cat(
            [
                batch_seq_tokens_tail_repr,
                torch.ones_like(batch_seq_tokens_tail_repr[..., :1]),
            ],
            dim=-1,
        )

        batch_joint_score = torch.einsum(
            "bxi, oij, byj -> boxy",
            batch_seq_tokens_head_repr,
            self.U,
            batch_seq_tokens_tail_repr,
        ).permute(0, 2, 3, 1)

        batch_normalized_joint_score = (
            torch.softmax(batch_joint_score, dim=-1)
            * batch_inputs["joint_label_matrix_mask"].unsqueeze(-1).float()
        )

        if not self.training:
            # tokens = [self.vocab.get_token_from_index(token, 'tokens') for token in batch_inputs['tokens'][batch_seq_tokens_lens]]
            # print("tokens: ", tokens)
            results["joint_label_preds"] = torch.argmax(
                batch_normalized_joint_score, dim=-1
            )

            # three step decoding happens in soft_joint_decoding func!
            separate_position_preds, ent_preds, rel_preds = self.soft_joint_decoding(
                batch_normalized_joint_score, batch_tokens, batch_seq_tokens_lens
            )

            results["all_separate_position_preds"] = separate_position_preds
            results["all_ent_preds"] = ent_preds
            results["all_rel_preds"] = rel_preds

            return results

        results["element_loss"] = self.element_loss(
            self.logit_dropout(
                batch_joint_score[batch_inputs["joint_label_matrix_mask"]]
            ),
            batch_inputs["joint_label_matrix"][batch_inputs["joint_label_matrix_mask"]],
        )

        batch_symmetric_normalized_joint_score = batch_normalized_joint_score[
            ..., self.symmetric_label
        ]
        results["symmetric_loss"] = (
            torch.abs(
                batch_symmetric_normalized_joint_score
                - batch_symmetric_normalized_joint_score.transpose(1, 2)
            )
            .sum(dim=-1)[batch_inputs["joint_label_matrix_mask"]]
            .mean()
        )

        batch_rel_normalized_joint_score = torch.max(
            batch_normalized_joint_score[..., self.rel_label], dim=-1
        ).values
        batch_diag_ent_normalized_joint_score = (
            torch.max(
                batch_normalized_joint_score[..., self.ent_label].diagonal(0, 1, 2),
                dim=1,
            )
            .values.unsqueeze(-1)
            .expand_as(batch_rel_normalized_joint_score)
        )

        results["implication_loss"] = (
            torch.relu(
                batch_rel_normalized_joint_score - batch_diag_ent_normalized_joint_score
            ).sum(dim=2)
            + torch.relu(
                batch_rel_normalized_joint_score.transpose(1, 2)
                - batch_diag_ent_normalized_joint_score
            ).sum(dim=2)
        )[batch_inputs["joint_label_matrix_mask"][..., 0]].mean()

        # relation_entities = batch_normalized_joint_score[..., self.ent_label[1]].diagonal(0, 1, 2)
        relation_entity_mask = batch_inputs["joint_label_matrix"].diagonal(0, 1, 2)
        relation_entity_mask = torch.eq(relation_entity_mask, self.ent_label[1])

        batch_row_subject_normalized_joint_score = torch.max(
            batch_normalized_joint_score[..., self.rel_label[0]], dim=-1
        ).values
        batch_column_subject_normalized_joint_score = torch.max(
            batch_normalized_joint_score.transpose(1, 2)[..., self.rel_label[0]], dim=-1
        ).values
        batch_row_object_normalized_joint_score = torch.max(
            batch_normalized_joint_score[..., self.rel_label[1]], dim=-1
        ).values
        batch_column_object_normalized_joint_score = torch.max(
            batch_normalized_joint_score.transpose(1, 2)[..., self.rel_label[1]], dim=-1
        ).values

        results["triple_loss"] = (
            (
                torch.relu(
                    batch_row_object_normalized_joint_score
                    - batch_row_subject_normalized_joint_score
                )
                + torch.relu(
                    batch_column_object_normalized_joint_score
                    - batch_column_subject_normalized_joint_score
                )
            )
            / 2
        )[relation_entity_mask].mean()

        return results

    def hard_joint_decoding(self, batch_normalized_joint_score, batch_seq_tokens_lens):
        """hard_joint_decoding extracts entity and relaition at the same time,
        and consider the interconnection of entity and relation.

        Args:
            batch_normalized_joint_score (tensor): batch joint pred
            batch_seq_tokens_lens (list): batch sequence length

        Returns:
            tuple: predicted entity and relation
        """
        separate_position_preds = []
        ent_preds = []
        rel_preds = []

        joint_label_n = self.vocab.get_vocab_size("ent_rel_id")
        batch_joint_pred = (
            torch.argmax(batch_normalized_joint_score, dim=-1).cpu().numpy()
        )
        ent_label = np.append(self.ent_label.cpu().numpy(), self.none_idx)
        rel_label = np.append(self.rel_label.cpu().numpy(), self.none_idx)

        for idx, seq_len in enumerate(batch_seq_tokens_lens):
            separate_position_preds.append([])
            ent_pred = {}
            rel_pred = {}
            ents = []
            joint_pred = batch_joint_pred[idx]
            ent_pos = [0] * seq_len
            for l in range(self.max_span_length, 0, -1):
                for st in range(0, seq_len - l + 1):
                    pred_cnt = np.array([0] * joint_label_n)
                    if any(ent_pos[st : st + l]):
                        continue
                    for i in range(st, st + l):
                        for j in range(st, st + l):
                            pred_cnt[joint_pred[i][j]] += 1
                    pred = int(ent_label[np.argmax(pred_cnt[ent_label])])
                    pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")

                    if pred_label == "None":
                        continue

                    ents.append((st, st + l))
                    for i in range(st, st + l):
                        ent_pos[i] = 1
                    ent_pred[(st, st + l)] = pred_label

            for idx1 in range(len(ents)):
                for idx2 in range(len(ents)):
                    if idx1 == idx2:
                        continue
                    pred_cnt = np.array([0] * joint_label_n)
                    for i in range(ents[idx1][0], ents[idx1][1]):
                        for j in range(ents[idx2][0], ents[idx2][1]):
                            pred_cnt[joint_pred[i][j]] += 1
                    pred = int(rel_label[np.argmax(pred_cnt[rel_label])])
                    pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")
                    h = ents[idx1][1] - ents[idx1][0]
                    w = ents[idx2][1] - ents[idx2][0]
                    if pred_label == "None":
                        continue
                    rel_pred[(ents[idx1], ents[idx2])] = pred_label

            ent_preds.append(ent_pred)
            rel_preds.append(rel_pred)

        return separate_position_preds, ent_preds, rel_preds

    def soft_joint_decoding(
        self, batch_normalized_joint_score, batch_tokens, batch_seq_tokens_lens
    ):
        """soft_joint_decoding extracts entity and relation at the same time,
        and consider the interconnection of entity and relation. This is used for measuring the ability of
         joint table filling model in generating Open-format extractions.

        Args:
            batch_normalized_joint_score (tensor): batch normalized joint score
            batch_seq_tokens_lens (list): batch sequence length

        Returns:
            tuple: predicted entity and relation
        """
        separate_position_preds = []
        ent_preds = []
        rel_preds = []

        batch_normalized_joint_score = batch_normalized_joint_score.cpu().numpy()
        symmetric_label = self.symmetric_label.cpu().numpy()
        ent_label = self.ent_label.cpu().numpy()
        rel_label = self.rel_label.cpu().numpy()

        for idx, seq_len in enumerate(batch_seq_tokens_lens):
            # print(" ".join([self.vocab.get_token_from_index(token.item(), 'tokens') for token in batch_tokens[idx][:seq_len]]))
            ent_pred = {}
            rel_pred = {}
            joint_score = batch_normalized_joint_score[idx][:seq_len, :seq_len, :]
            pred_label_tensors = copy.copy(joint_score)
            joint_score[..., symmetric_label] = (
                joint_score[..., symmetric_label]
                + joint_score[..., symmetric_label].transpose((1, 0, 2))
            ) / 2

            joint_score_feature = joint_score.reshape(seq_len, -1)
            transposed_joint_score_feature = joint_score.transpose((1, 0, 2)).reshape(
                seq_len, -1
            )
            separate_pos = (
                (
                    np.linalg.norm(
                        joint_score_feature[0 : seq_len - 1]
                        - joint_score_feature[1:seq_len],
                        axis=1,
                    )
                    + np.linalg.norm(
                        transposed_joint_score_feature[0 : seq_len - 1]
                        - transposed_joint_score_feature[1:seq_len],
                        axis=1,
                    )
                )
                * 0.5
                > self.separate_threshold
            ).nonzero()[0]
            separate_position_preds.append([pos.item() for pos in separate_pos])
            if len(separate_pos) > 0:
                spans = (
                    [(0, separate_pos[0].item() + 1)]
                    + [
                        (separate_pos[idx].item() + 1, separate_pos[idx + 1].item() + 1)
                        for idx in range(len(separate_pos) - 1)
                    ]
                    + [(separate_pos[-1].item() + 1, seq_len)]
                )
            else:
                spans = [(0, seq_len)]

            # merged_spans = self.merge_similar_spans(spans, joint_score)
            merged_spans = [(span,) for span in spans]
            ents = []
            index2span = {}
            for span in merged_spans:
                target_indices = []
                for sp in span:
                    target_indices += [idx for idx in range(sp[0], sp[1])]
                score = np.mean(
                    joint_score[target_indices, :, :][:, target_indices, :], axis=(0, 1)
                )
                if not (np.max(score[ent_label]) < score[self.none_idx]):
                    pred = ent_label[np.argmax(score[ent_label])].item()
                    pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")
                    ents.append(target_indices)
                    index2span[tuple(target_indices)] = span
                    ent_pred[span] = pred_label

            for ent1 in ents:
                for ent2 in ents:
                    if ent1 == ent2:
                        continue
                    score = np.mean(joint_score[ent1, :, :][:, ent2, :], axis=(0, 1))
                    if not (np.max(score[rel_label]) < score[self.none_idx]):
                        pred = rel_label[np.argmax(score[rel_label])].item()
                        pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")
                        rel_pred[(index2span[tuple(ent1)], index2span[tuple(ent2)])] = (
                            pred_label
                        )
                    elif (
                        ent_pred[index2span[tuple(ent1)]] == "Relation"
                        and ent_pred[index2span[tuple(ent2)]] == "Argument"
                    ) or (
                        ent_pred[index2span[tuple(ent1)]] == "Argument"
                        and ent_pred[index2span[tuple(ent2)]] == "Relation"
                    ):
                        joint_score_tensor = torch.from_numpy(
                            pred_label_tensors[ent1, :, :][:, ent2, :]
                        )
                        batch_joint_pred_sorted = torch.argsort(
                            joint_score_tensor, dim=-1, descending=True
                        )
                        most_possible_label = torch.argmax(joint_score_tensor, dim=-1)
                        # assert most_possible_label == batch_joint_pred_sorted[...,0]
                        second_possible_label = batch_joint_pred_sorted[..., 1]
                        subj_indices = torch.nonzero(most_possible_label == 3)
                        obj_indices = torch.nonzero(most_possible_label == 4)
                        if subj_indices.nelement() != 0 and obj_indices.nelement() == 0:
                            second_possible_label = second_possible_label[
                                (most_possible_label != 3).nonzero(as_tuple=True)
                            ]
                            second_possible_label = (second_possible_label * 1.0).mean()
                            if 2.7 < second_possible_label <= 3.3:
                                rel_pred[
                                    (index2span[tuple(ent1)], index2span[tuple(ent2)])
                                ] = "Subject"

                        elif (
                            obj_indices.nelement() != 0 and subj_indices.nelement() == 0
                        ):
                            second_possible_label = second_possible_label[
                                (most_possible_label != 4).nonzero(as_tuple=True)
                            ]
                            second_possible_label = (second_possible_label * 1.0).mean()
                            if 3.5 < second_possible_label <= 4:
                                rel_pred[
                                    (index2span[tuple(ent1)], index2span[tuple(ent2)])
                                ] = "Object"

                        else:
                            if 1 <= (ent2[0] - ent1[-1]) < 3:
                                second_possible_label = (
                                    (second_possible_label * 1.0).mean().item()
                                )
                                if ent_pred[index2span[tuple(ent1)]] == "Relation" or (
                                    ent_pred[index2span[tuple(ent2)]] == "Relation"
                                    and ent2[0] == seq_len - 6
                                ):
                                    # if 3.5 < second_possible_label <= 4:
                                    rel_pred[
                                        (
                                            index2span[tuple(ent1)],
                                            index2span[tuple(ent2)],
                                        )
                                    ] = "Object"
                                else:
                                    if 2.7 < second_possible_label <= 3.3:
                                        rel_pred[
                                            (
                                                index2span[tuple(ent1)],
                                                index2span[tuple(ent2)],
                                            )
                                        ] = "Subject"

            ent_preds.append(ent_pred)
            rel_preds.append(rel_pred)

        return separate_position_preds, ent_preds, rel_preds
