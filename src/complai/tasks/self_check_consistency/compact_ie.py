from __future__ import annotations

import io
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer

from complai.tasks.self_check_consistency.constituent_linking.inputs.dataset_readers.oie_reader_for_ent_rel_decoding import (
    OIE4ReaderForEntRelDecoding,
)
from complai.tasks.self_check_consistency.constituent_linking.inputs.datasets.dataset import (
    Dataset,
)
from complai.tasks.self_check_consistency.constituent_linking.inputs.fields.map_token_field import (
    MapTokenField,
)
from complai.tasks.self_check_consistency.constituent_linking.inputs.fields.raw_token_field import (
    RawTokenField,
)
from complai.tasks.self_check_consistency.constituent_linking.inputs.fields.token_field import (
    TokenField,
)
from complai.tasks.self_check_consistency.constituent_linking.inputs.instance import (
    Instance,
)
from complai.tasks.self_check_consistency.constituent_linking.inputs.vocabulary import (
    Vocabulary,
)
from complai.tasks.self_check_consistency.constituent_linking.models.joint_decoding.joint_decoder import (
    EntRelJointDecoder,
)
from complai.tasks.self_check_consistency.constituent_linking.models.relation_decoding.relation_decoder import (
    RelDecoder,
)
from complai.tasks.self_check_consistency.constituent_linking.utils.prediction_outputs import (
    print_extractions_jsonl_format,
)


sys.path.append(str(Path(__file__).parent))


def get_sentence_dicts(sentence: str, sent_id: Any) -> list[dict[str, Any]]:
    sentence = sentence.replace("\n", "")
    return [
        {
            "sentence": sentence
            + " [unused1] [unused2] [unused3] [unused4] [unused5] [unused6]",
            "sentId": sent_id,
            "entityMentions": list(),
            "relationMentions": list(),
            "extractionMentions": list(),
        }
    ]


def add_joint_label(ext: dict[str, Any], ent_rel_id: dict[str, Any]) -> None:
    """add_joint_label add joint labels for sentences."""
    none_id = ent_rel_id["None"]
    sentence_length = len(ext["sentText"].split(" "))
    entity_label_matrix = [
        [none_id for _ in range(sentence_length)] for _ in range(sentence_length)
    ]
    relation_label_matrix = [
        [none_id for _ in range(sentence_length)] for _ in range(sentence_length)
    ]
    label_matrix = [
        [none_id for _ in range(sentence_length)] for _ in range(sentence_length)
    ]
    ent2offset = {}
    for ent in ext["entityMentions"]:
        ent2offset[ent["emId"]] = ent["span_ids"]
        try:
            for i in ent["span_ids"]:
                for j in ent["span_ids"]:
                    entity_label_matrix[i][j] = ent_rel_id[ent["label"]]
        except:  #  noqa E772
            print("span ids: ", sentence_length, ent["span_ids"], ext)
            sys.exit(1)
    ext["entityLabelMatrix"] = entity_label_matrix
    for rel in ext["relationMentions"]:
        arg1_span = ent2offset[rel["arg1"]["emId"]]
        arg2_span = ent2offset[rel["arg2"]["emId"]]

        for i in arg1_span:
            for j in arg2_span:
                # to be consistent with the linking model
                relation_label_matrix[i][j] = ent_rel_id[rel["label"]] - 2
                relation_label_matrix[j][i] = ent_rel_id[rel["label"]] - 2
                label_matrix[i][j] = ent_rel_id[rel["label"]]
                label_matrix[j][i] = ent_rel_id[rel["label"]]
    ext["relationLabelMatrix"] = relation_label_matrix
    ext["jointLabelMatrix"] = label_matrix


def tokenize_sentences(ext: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    wordpiece_tokens = [cls]

    wordpiece_tokens_index = []
    cur_index = len(wordpiece_tokens)
    # for token in ext['sentText'].split(' '):
    for token in ext["sentence"].split(" "):
        # print(token)
        tokenized_token = list(tokenizer.tokenize(token))
        # print(tokenized_token)
        wordpiece_tokens.extend(tokenized_token)
        wordpiece_tokens_index.append([cur_index, cur_index + len(tokenized_token)])
        cur_index += len(tokenized_token)
    wordpiece_tokens.append(sep)

    wordpiece_segment_ids = [1] * (len(wordpiece_tokens))
    return {
        "sentId": ext["sentId"],
        "sentText": ext["sentence"],
        "entityMentions": ext["entityMentions"],
        "relationMentions": ext["relationMentions"],
        "extractionMentions": ext["extractionMentions"],
        "wordpieceSentText": " ".join(wordpiece_tokens),
        "wordpieceTokensIndex": wordpiece_tokens_index,
        "wordpieceSegmentIds": wordpiece_segment_ids,
    }


def process(
    fin: io.StringIO, fout: Any, tokenizer: Any, ent_rel_file: dict[str, Any]
) -> None:
    extractions_list = []

    ent_rel_id = ent_rel_file["id"]
    sentId = 0
    for line in fin:
        sentId += 1
        exts = get_sentence_dicts(line, sentId)
        for ext in exts:
            ext_dict = tokenize_sentences(ext, tokenizer)
            add_joint_label(ext_dict, ent_rel_id)
            extractions_list.append(ext_dict)
            fout.write(json.dumps(ext_dict))
            fout.write("\n")


logger = logging.getLogger(__name__)


def step(
    ent_model: EntRelJointDecoder,
    rel_model: RelDecoder,
    batch_inputs: Any,
    main_vocab: Any,
    device: int,
) -> list[Any] | tuple:
    batch_inputs["tokens"] = torch.LongTensor(batch_inputs["tokens"])
    batch_inputs["entity_label_matrix"] = torch.LongTensor(
        batch_inputs["entity_label_matrix"]
    )
    batch_inputs["entity_label_matrix_mask"] = torch.BoolTensor(
        batch_inputs["entity_label_matrix_mask"]
    )
    batch_inputs["relation_label_matrix"] = torch.LongTensor(
        batch_inputs["relation_label_matrix"]
    )
    batch_inputs["relation_label_matrix_mask"] = torch.BoolTensor(
        batch_inputs["relation_label_matrix_mask"]
    )
    batch_inputs["wordpiece_tokens"] = torch.LongTensor(
        batch_inputs["wordpiece_tokens"]
    )
    batch_inputs["wordpiece_tokens_index"] = torch.LongTensor(
        batch_inputs["wordpiece_tokens_index"]
    )
    batch_inputs["wordpiece_segment_ids"] = torch.LongTensor(
        batch_inputs["wordpiece_segment_ids"]
    )

    batch_inputs["joint_label_matrix"] = torch.LongTensor(
        batch_inputs["joint_label_matrix"]
    )
    batch_inputs["joint_label_matrix_mask"] = torch.BoolTensor(
        batch_inputs["joint_label_matrix_mask"]
    )
    if device > -1:
        batch_inputs["tokens"] = batch_inputs["tokens"].cuda(
            device=device, non_blocking=True
        )
        batch_inputs["entity_label_matrix"] = batch_inputs["entity_label_matrix"].cuda(
            device=device, non_blocking=True
        )
        batch_inputs["entity_label_matrix_mask"] = batch_inputs[
            "entity_label_matrix_mask"
        ].cuda(device=device, non_blocking=True)
        batch_inputs["relation_label_matrix"] = batch_inputs[
            "relation_label_matrix"
        ].cuda(device=device, non_blocking=True)
        batch_inputs["relation_label_matrix_mask"] = batch_inputs[
            "relation_label_matrix_mask"
        ].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_tokens"] = batch_inputs["wordpiece_tokens"].cuda(
            device=device, non_blocking=True
        )
        batch_inputs["wordpiece_tokens_index"] = batch_inputs[
            "wordpiece_tokens_index"
        ].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_segment_ids"] = batch_inputs[
            "wordpiece_segment_ids"
        ].cuda(device=device, non_blocking=True)

        batch_inputs["joint_label_matrix"] = batch_inputs["joint_label_matrix"].cuda(
            device=device, non_blocking=True
        )
        batch_inputs["joint_label_matrix_mask"] = batch_inputs[
            "joint_label_matrix_mask"
        ].cuda(device=device, non_blocking=True)

    ent_outputs = ent_model(batch_inputs, rel_model, main_vocab)
    batch_outputs = []
    if not ent_model.training and not rel_model.training:
        # entities
        for sent_idx in range(len(batch_inputs["tokens_lens"])):
            sent_output = dict()
            sent_output["tokens"] = batch_inputs["tokens"][sent_idx].cpu().numpy()
            sent_output["span2ent"] = batch_inputs["span2ent"][sent_idx]
            sent_output["span2rel"] = batch_inputs["span2rel"][sent_idx]
            sent_output["seq_len"] = batch_inputs["tokens_lens"][sent_idx]
            sent_output["entity_label_matrix"] = (
                batch_inputs["entity_label_matrix"][sent_idx].cpu().numpy()
            )
            sent_output["entity_label_preds"] = (
                ent_outputs["entity_label_preds"][sent_idx].cpu().numpy()
            )
            sent_output["separate_positions"] = batch_inputs["separate_positions"][
                sent_idx
            ]
            sent_output["all_separate_position_preds"] = ent_outputs[
                "all_separate_position_preds"
            ][sent_idx]
            sent_output["all_ent_preds"] = ent_outputs["all_ent_preds"][sent_idx]
            sent_output["all_rel_preds"] = ent_outputs["all_rel_preds"][sent_idx]
            batch_outputs.append(sent_output)
        return batch_outputs

    return ent_outputs["element_loss"], ent_outputs["symmetric_loss"]


def run_model(
    dataset: Dataset,
    ent_model: EntRelJointDecoder,
    rel_model: RelDecoder,
    out_file: io.StringIO,
    device: int,
    test_batch_size: int,
    conjunctions_file: str,
) -> None:
    ent_model.zero_grad()
    rel_model.zero_grad()

    all_outputs: list = []
    # TODO transform input line to batch
    ent_model.eval()
    rel_model.eval()
    for idx, batch in dataset.get_batch("test", test_batch_size, None):
        logger.info("{} processed".format(idx + 1))
        with torch.no_grad():
            batch_outputs = step(ent_model, rel_model, batch, dataset.vocab, device)
        all_outputs.extend(batch_outputs)
    print_extractions_jsonl_format(
        conjunctions_file, all_outputs, dataset.vocab, out_file
    )


class CompactFactsOpenInformationExtraction:
    def __init__(
        self,
        *,
        seed: int = 42,
        # default is using the cpu
        device: int = -1,
        embedding_model: Literal["bert", "pretrained"] = "bert",
        max_sent_len: int = 200,
        max_wordpiece_len: int = 512,
        # ent_rel_file: str = str(
        #     Path(__file__).parent / "constituent_linking" / "data" / "ent_rel_file.json"
        # ),
        # rel_file: str = str(
        #     Path(__file__).parent / "constituent_linking" / "data" / "rel_file.json"
        # ),
        hf_model_repo: str = "compl-ai/compact_ie",
        bert_model_name: str = "bert-base-uncased",
        pretrained_model_name: str = "",
        max_span_length: int = 10,
        separate_threshold: float = 1.0,
        mlp_hidden_size: int = 150,
        dropout: float = 0.4,
        logit_dropout: float = 0.2,
        fine_tune: bool = False,
        bert_output_size: int = 0,
        bert_dropout: float = 0.0,
        test_batch_size: int = 32,
        # conjunctions_file: str = str(
        #     Path(__file__).parent
        #     / "constituent_linking"
        #     / "data"
        #     / "carb_test_conjunctions.txt"
        # ),
    ) -> None:
        self.device = device
        self.test_batch_size = test_batch_size
        self.hf_model_repo = hf_model_repo

        # compact_ie_dir = str(Path(__file__).parent / "constituent_linking" / "data")
        # constituent_vocab = os.path.join(compact_ie_dir, "constituent_vocabulary.pkl")
        # relation_vocab = os.path.join(compact_ie_dir, "relation_vocabulary.pkl")
        # constituent_model_path = os.path.join(
        #     compact_ie_dir, "constituent_model_weights"
        # )
        # relation_model_path = os.path.join(compact_ie_dir, "linking_model_weights")

        # Download all required files from HF Hub
        constituent_vocab = hf_hub_download(
            repo_id=hf_model_repo, filename="vocabs/constituent_vocabulary.pkl"
        )
        relation_vocab = hf_hub_download(
            repo_id=hf_model_repo, filename="vocabs/relation_vocabulary.pkl"
        )
        constituent_model_path = hf_hub_download(
            repo_id=hf_model_repo,
            filename="models/constituent/constituent_model_weights",
        )
        relation_model_path = hf_hub_download(
            repo_id=hf_model_repo, filename="models/linking/linking_model_weights"
        )
        ent_rel_file = hf_hub_download(
            repo_id=hf_model_repo, filename="relation/ent_rel_file.json"
        )
        rel_file = hf_hub_download(
            repo_id=hf_model_repo, filename="relation/rel_file.json"
        )
        self.conjunctions_file = hf_hub_download(
            repo_id=hf_model_repo, filename="conjunctions/carb_test_conjunctions.txt"
        )

        # set random seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device > -1 and not torch.cuda.is_available():
            logger.error("config conflicts: no gpu available, use cpu for training.")
            device = -1
        if device > -1:
            torch.cuda.manual_seed(seed)

        # define fields
        tokens = TokenField("tokens", "tokens", "tokens", True)
        separate_positions = RawTokenField("separate_positions", "separate_positions")
        span2ent = MapTokenField("span2ent", "ent_rel_id", "span2ent", False)
        span2rel = MapTokenField("span2rel", "ent_rel_id", "span2rel", False)
        entity_label_matrix = RawTokenField(
            "entity_label_matrix", "entity_label_matrix"
        )
        relation_label_matrix = RawTokenField(
            "relation_label_matrix", "relation_label_matrix"
        )
        joint_label_matrix = RawTokenField("joint_label_matrix", "joint_label_matrix")
        wordpiece_tokens = TokenField(
            "wordpiece_tokens", "wordpiece", "wordpiece_tokens", False
        )
        wordpiece_tokens_index = RawTokenField(
            "wordpiece_tokens_index", "wordpiece_tokens_index"
        )
        wordpiece_segment_ids = RawTokenField(
            "wordpiece_segment_ids", "wordpiece_segment_ids"
        )
        fields = [
            tokens,
            separate_positions,
            span2ent,
            span2rel,
            entity_label_matrix,
            relation_label_matrix,
            joint_label_matrix,
        ]

        if embedding_model in ["bert", "pretrained"]:
            fields.extend(
                [wordpiece_tokens, wordpiece_tokens_index, wordpiece_segment_ids]
            )

        self.fields = fields

        # define counter and vocabulary
        counter: defaultdict = defaultdict(lambda: defaultdict(int))
        self.counter = counter
        vocab_ent = Vocabulary()

        # define dataset reader
        max_len = {"tokens": max_sent_len, "wordpiece_tokens": max_wordpiece_len}
        self.max_len = max_len
        self.ent_rel_file = json.load(open(ent_rel_file, "r", encoding="utf-8"))
        rel_file = json.load(open(rel_file, "r", encoding="utf-8"))
        pretrained_vocab = {"ent_rel_id": self.ent_rel_file["id"]}
        if embedding_model == "bert":
            # bert_base_uncased_dir = (
            #     Path(__file__).parent
            #     / "constituent_linking"
            #     / "data"
            #     / "models"
            #     / "bert"
            # )
            # Load tokenizer from HF repo subfolder
            tokenizer = BertTokenizer.from_pretrained(
                hf_model_repo, subfolder="models/bert-base-uncased"
            )
            logger.info("Load bert tokenizer from HF successfully.")
            pretrained_vocab["wordpiece"] = tokenizer.get_vocab()
        elif embedding_model == "pretrained":
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
            logger.info("Load {} tokenizer successfully.".format(pretrained_model_name))
            pretrained_vocab["wordpiece"] = tokenizer.get_vocab()

        self.tokenizer = tokenizer
        self.pretrained_vocab = pretrained_vocab

        vocab_ent = Vocabulary.load(constituent_vocab)
        vocab_rel = Vocabulary.load(relation_vocab)
        self.vocab_ent = vocab_ent
        # # separate models for constituent generation and linking
        # ent_model = EntRelJointDecoder(cfg=cfg, vocab=vocab_ent, ent_rel_file=ent_rel_file, rel_file=rel_file)
        # rel_model = RelDecoder(cfg=cfg, vocab=vocab_rel, ent_rel_file=rel_file)
        # separate models for constituent generation and linking

        # Initialize models with HF repo path for BERT model
        # bert_model_path = (
        #     f"{hf_model_repo}" if embedding_model == "bert" else bert_model_name
        # )

        ent_model = EntRelJointDecoder(
            vocab=vocab_ent,
            ent_rel_file=self.ent_rel_file,
            rel_file=rel_file,
            max_span_length=max_span_length,
            device=device,
            separate_threshold=separate_threshold,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
            logit_dropout=logit_dropout,
            embedding_model=embedding_model,
            bert_model_name=bert_model_name,
            # bert_model_name=hf_model_repo,
            pretrained_model_name=pretrained_model_name,
            fine_tune=fine_tune,
            bert_output_size=bert_output_size,
            bert_dropout=bert_dropout,
        )
        rel_model = RelDecoder(
            vocab=vocab_rel,
            ent_rel_file=self.ent_rel_file,
            max_span_length=max_span_length,
            device=device,
            logit_dropout=logit_dropout,
            bert_model_name=bert_model_name,
            # bert_model_name=hf_model_repo,
            fine_tune=fine_tune,
            bert_output_size=bert_output_size,
            bert_dropout=bert_dropout,
        )

        # main bert-based model
        if os.path.exists(constituent_model_path):
            state_dict = torch.load(  # nosec B614
                open(constituent_model_path, "rb"),
                map_location=lambda storage, _: storage,
                weights_only=False,
            )
            ent_model.load_state_dict(state_dict, strict=False)
            # ent_model.embedding_model.bert_encoder.bert_model = BertModel.from_pretrained(  # type: ignore
            #         'bert-base-uncased',
            #         state_dict=state_dict
            #         )
            # print("constituent model loaded")
        else:
            raise FileNotFoundError
        if os.path.exists(relation_model_path):
            state_dict = torch.load(  # nosec B614
                open(relation_model_path, "rb"),
                map_location=lambda storage, _: storage,
                # weights_only=False
            )
            rel_model.load_state_dict(state_dict, strict=False)
            # rel_model.embedding_model.bert_encoder.bert_model = BertModel.from_pretrained(
            #         'bert-base-uncased',
            #         state_dict=state_dict
            #         )
            # print("linking model loaded")
        else:
            raise FileNotFoundError
        logger.info("Loading best training models successfully for testing.")

        if device > -1:
            ent_model.cuda(device=device)
            rel_model.cuda(device=device)

        self.entity_model = ent_model
        self.relation_model = rel_model

    def process_sentences(self, sentences: list[str]) -> list[dict[str, str]]:
        # try:
        raw = io.StringIO()

        for line in sentences:
            raw.write(line)
            raw.write("\n")

        logger.info("Tokenizing input")
        raw.seek(0)
        formatted = io.StringIO()
        process(raw, formatted, self.tokenizer, self.ent_rel_file)
        raw.close()
        formatted.seek(0)
        oie_test_reader = OIE4ReaderForEntRelDecoding(formatted, False, self.max_len)

        # define instance (data sets)
        test_instance = Instance(self.fields)

        logger.info("Formatting input")
        oie_dataset = Dataset("OIE4")
        oie_dataset.add_instance(
            "test", test_instance, oie_test_reader, is_count=True, is_train=False
        )

        min_count = {"tokens": 1}
        no_pad_namespace = ["ent_rel_id"]
        no_unk_namespace = ["ent_rel_id"]
        contain_pad_namespace = {"wordpiece": self.tokenizer.pad_token}
        contain_unk_namespace = {"wordpiece": self.tokenizer.unk_token}
        oie_dataset.build_dataset(
            vocab=self.vocab_ent,
            counter=self.counter,
            min_count=min_count,
            pretrained_vocab=self.pretrained_vocab,
            no_pad_namespace=no_pad_namespace,
            no_unk_namespace=no_unk_namespace,
            contain_pad_namespace=contain_pad_namespace,
            contain_unk_namespace=contain_unk_namespace,
        )
        wo_padding_namespace = ["separate_positions", "span2ent", "span2rel"]
        oie_dataset.set_wo_padding_namespace(wo_padding_namespace=wo_padding_namespace)

        logger.info("Processing input")
        jsonl = io.StringIO()

        run_model(
            oie_dataset,
            self.entity_model,
            self.relation_model,
            jsonl,
            self.device,
            self.test_batch_size,
            self.conjunctions_file,
        )

        jsonl.seek(0)
        response = f"[{','.join(jsonl.readlines())}]"
        formatted.close()
        jsonl.close()
        # except Exception as e:
        #     self.send_error(500, message=str(e))
        return json.loads(response)
