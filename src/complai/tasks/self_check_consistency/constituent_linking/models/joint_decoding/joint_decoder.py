import logging

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from complai.tasks.self_check_consistency.constituent_linking.models.embedding_models.bert_embedding_model import (
    BertEmbedModel,
)
from complai.tasks.self_check_consistency.constituent_linking.models.embedding_models.pretrained_embedding_model import (
    PretrainedEmbedModel,
)
from complai.tasks.self_check_consistency.constituent_linking.modules.token_embedders.bert_encoder import (
    BertLinear,
)


logger = logging.getLogger(__name__)


class EntRelJointDecoder(nn.Module):
    Argument_START_NER = "<START=Argument>".lower()
    Argument_END_NER = "<END=Argument>".lower()
    Relation_START_NER = "<START=Relation>".lower()
    Relation_END_NER = "<END=Relation>".lower()

    def __init__(
        self,
        vocab,
        ent_rel_file,
        rel_file,
        max_span_length,
        device,
        separate_threshold,
        mlp_hidden_size,
        dropout,
        logit_dropout,
        embedding_model,
        bert_model_name,
        pretrained_model_name,
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
        self.auto_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.cls = self.auto_tokenizer.cls_token
        self.sep = self.auto_tokenizer.sep_token
        self.rel_file = rel_file
        self.add_marker_tokens()
        self.vocab = vocab
        self.max_span_length = max_span_length
        self.activation = nn.GELU()
        self.device = device
        self.separate_threshold = separate_threshold

        if embedding_model == "bert":
            self.embedding_model = BertEmbedModel(
                bert_model_name, fine_tune, bert_output_size, bert_dropout, vocab
            )
        elif embedding_model == "pretrained":
            self.embedding_model = PretrainedEmbedModel(
                pretrained_model_name, fine_tune, bert_output_size, bert_dropout, vocab
            )
        self.encoder_output_size = self.embedding_model.get_hidden_size()

        self.head_mlp = BertLinear(
            input_size=self.encoder_output_size,
            output_size=mlp_hidden_size,
            activation=self.activation,
            dropout=dropout,
        )
        self.tail_mlp = BertLinear(
            input_size=self.encoder_output_size,
            output_size=mlp_hidden_size,
            activation=self.activation,
            dropout=dropout,
        )

        self.U = nn.Parameter(
            torch.FloatTensor(
                self.vocab.get_vocab_size("ent_rel_id"),
                mlp_hidden_size + 1,
                mlp_hidden_size + 1,
            )
        )
        self.U.data.zero_()

        if logit_dropout > 0:
            self.logit_dropout = nn.Dropout(p=logit_dropout)
        else:
            self.logit_dropout = lambda x: x

        self.none_idx = self.vocab.get_token_index("None", "ent_rel_id")

        self.symmetric_label = torch.LongTensor(ent_rel_file["symmetric"])
        self.asymmetric_label = torch.LongTensor(ent_rel_file["asymmetric"])
        self.ent_label = torch.LongTensor(ent_rel_file["entity"])
        self.rel_label = torch.LongTensor(ent_rel_file["relation"])
        # self.rel_label = torch.LongTensor([r - 2 for r in ent_rel_file["relation"]])
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

    def add_marker_tokens(self):
        new_tokens = ["<START>", "<END>"]
        for label in self.rel_file["entity_text"]:
            new_tokens.append("<START=%s>" % label)
            new_tokens.append("<END=%s>" % label)
        self.auto_tokenizer.add_tokens(new_tokens)
        # print('# vocab after adding markers: %d'%len(tokenizer))

    def forward(self, batch_inputs, rel_model, dataset_vocab):
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
            results["entity_label_preds"] = torch.argmax(
                batch_normalized_joint_score, dim=-1
            )

            separate_position_preds, ent_preds, rel_preds = self.soft_joint_decoding(
                batch_normalized_joint_score,
                rel_model,
                batch_tokens,
                batch_seq_tokens_lens,
                dataset_vocab,
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

    def soft_joint_decoding(
        self,
        batch_normalized_entity_score,
        rel_model,
        batch_tokens,
        batch_seq_tokens_lens,
        dataset_vocab,
    ):
        separate_position_preds = []
        ent_preds = []
        rel_preds = []

        batch_normalized_entity_score = batch_normalized_entity_score.cpu().numpy()
        ent_label = self.ent_label.cpu().numpy()
        rel_label = self.rel_label.cpu().numpy()

        for idx, seq_len in enumerate(batch_seq_tokens_lens):
            # joint_rel_score = relation_matrix[idx][:seq_len, :seq_len, :]
            tokens = [
                dataset_vocab.get_token_from_index(token.item(), "tokens")
                for token in batch_tokens[idx][:seq_len]
            ]

            ent_pred = {}
            rel_pred = {}
            entity_score = batch_normalized_entity_score[idx][:seq_len, :seq_len, :]
            entity_score = (entity_score + entity_score.transpose((1, 0, 2))) / 2

            entity_score_feature = entity_score.reshape(seq_len, -1)
            transposed_entity_score_feature = entity_score.transpose((1, 0, 2)).reshape(
                seq_len, -1
            )
            separate_pos = (
                (
                    np.linalg.norm(
                        entity_score_feature[0 : seq_len - 1]
                        - entity_score_feature[1:seq_len],
                        axis=1,
                    )
                    + np.linalg.norm(
                        transposed_entity_score_feature[0 : seq_len - 1]
                        - transposed_entity_score_feature[1:seq_len],
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

            merged_spans = [(span,) for span in spans]
            ents = []
            relations = []
            arguments = []
            index2span = {}
            for span in merged_spans:
                target_indices = []
                for sp in span:
                    target_indices += [idx for idx in range(sp[0], sp[1])]
                score = np.mean(
                    entity_score[target_indices, :, :][:, target_indices, :],
                    axis=(0, 1),
                )
                if not (np.max(score[ent_label]) < score[self.none_idx]):
                    pred = ent_label[np.argmax(score[ent_label])].item()
                    pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")
                    if pred_label == "Relation":
                        relations.append(target_indices)
                    else:
                        arguments.append(target_indices)
                    ents.append(target_indices)
                    index2span[tuple(target_indices)] = span
                    ent_pred[span] = pred_label

            # relation decode begins
            for rel in relations:
                subj_found = False
                obj_found = False
                # if rel[-1] < seq_len - 6:
                sorted_arguments = sorted(arguments, key=lambda a: abs(a[0] - rel[0]))
                sorted_indices = [arguments.index(arg) for arg in sorted_arguments]
                argument_start_ids = [arg[0] for arg in sorted_arguments]
                argument_end_ids = [arg[-1] for arg in sorted_arguments]
                relation_indices = []
                argument_indices = []
                wordpiece_tokens = [self.cls]
                for i, token in enumerate(tokens):
                    if i == rel[0]:
                        relation_indices.append(len(wordpiece_tokens))
                        wordpiece_tokens.append(self.Relation_START_NER)
                    if i in argument_start_ids:
                        argument_indices.append(len(wordpiece_tokens))
                        wordpiece_tokens.append(self.Argument_START_NER)

                    tokenized_token = list(self.auto_tokenizer.tokenize(token))
                    wordpiece_tokens.extend(tokenized_token)
                    if i == rel[-1]:
                        wordpiece_tokens.append(self.Relation_END_NER)
                    if i in argument_end_ids:
                        wordpiece_tokens.append(self.Argument_END_NER)

                wordpiece_tokens.append(self.sep)
                wordpiece_segment_ids = [1] * (len(wordpiece_tokens))
                wordpiece_tokens = [
                    rel_model.vocab.get_token_index(token, "wordpiece")
                    for token in wordpiece_tokens
                ]
                rel_input = {
                    "wordpiece_tokens": torch.LongTensor([wordpiece_tokens]),
                    "relation_ids": torch.LongTensor(
                        [relation_indices * len(argument_indices)]
                    ),
                    "argument_ids": torch.LongTensor([argument_indices]),
                    "label_ids_mask": torch.LongTensor([[1] * len(argument_indices)]),
                    "wordpiece_segment_ids": torch.LongTensor([wordpiece_segment_ids]),
                }
                if self.device > -1:
                    for k in rel_input:
                        rel_input[k] = rel_input[k].cuda(0)

                output = rel_model(rel_input)
                output = output["label_preds"][0].cpu().numpy()
                sorted_output_labels = [output[i] for i in sorted_indices]
                assert len(argument_start_ids) == len(output)
                prev_subj = 0
                prev_obj = 0
                for idx, label_id in enumerate(sorted_output_labels):
                    if label_id == 0 and subj_found and obj_found:
                        break

                    pred_label = "None"
                    pred_t_label = "None"
                    score = np.mean(
                        entity_score[rel, :, :][:, sorted_arguments[idx], :],
                        axis=(0, 1),
                    )
                    score_t = np.mean(
                        entity_score[sorted_arguments[idx], :, :][:, rel, :],
                        axis=(0, 1),
                    )
                    if not (np.max(score[rel_label]) < score[self.none_idx]) or not (
                        np.max(score_t[rel_label]) < score_t[self.none_idx]
                    ):
                        pred = rel_label[np.argmax(score[rel_label])].item()
                        pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")

                        pred = rel_label[np.argmax(score_t[rel_label])].item()
                        pred_t_label = self.vocab.get_token_from_index(
                            pred, "ent_rel_id"
                        )

                    # to handle object less extractions
                    if label_id == 1 and sorted_arguments[idx][0] > rel[-1]:
                        obj_found = True
                        if (pred_label == "Object" or pred_t_label == "Object") and (
                            not obj_found
                            or (
                                prev_obj != 0
                                and prev_obj + 1
                                <= sorted_arguments[idx][0]
                                <= prev_obj + 3
                            )
                        ):
                            rel_pred[
                                (
                                    index2span[tuple(rel)],
                                    index2span[tuple(sorted_arguments[idx])],
                                )
                            ] = "Object"
                            prev_obj = sorted_arguments[idx][-1]
                        continue

                    # just added (maybe need to be deleted)
                    if label_id == 2 and sorted_arguments[idx][0] < rel[0]:
                        if (pred_label == "Subject" or pred_t_label == "Subject") and (
                            not subj_found
                            or (
                                prev_subj != 0
                                and prev_subj - 1 == sorted_arguments[idx][-1]
                            )
                        ):
                            rel_pred[
                                (
                                    index2span[tuple(rel)],
                                    index2span[tuple(sorted_arguments[idx])],
                                )
                            ] = "Subject"
                            subj_found = True
                            prev_subj = sorted_arguments[idx][0]
                        continue

                    if label_id == 1 and (
                        not subj_found
                        or (
                            prev_subj != 0
                            and sorted_arguments[idx][-1] == prev_subj - 1
                        )
                    ):
                        rel_pred[
                            (
                                index2span[tuple(rel)],
                                index2span[tuple(sorted_arguments[idx])],
                            )
                        ] = "Subject"
                        subj_found = True
                        prev_subj = sorted_arguments[idx][0]

                    elif label_id == 2 and (
                        not obj_found
                        or (prev_obj != 0 and prev_obj + 1 == sorted_arguments[idx][0])
                    ):
                        rel_pred[
                            (
                                index2span[tuple(rel)],
                                index2span[tuple(sorted_arguments[idx])],
                            )
                        ] = "Object"
                        obj_found = True
                        prev_obj = sorted_arguments[idx][-1]

            ent_preds.append(ent_pred)
            rel_preds.append(rel_pred)

        return separate_position_preds, ent_preds, rel_preds
