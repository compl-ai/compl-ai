import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from .constants import BAD, FUNC, GOOD
from .dataset import CodeDataset
from .utils import load_model, set_seed


class LossDict:
    def __init__(self):
        self.d = OrderedDict()
        self.d["func"] = list()
        self.d["pos"] = list()
        self.d["neg"] = list()

    def step(self, other):
        for k in other.d:
            self.d[k] += other.d[k]

    def pretty_print(self, args):
        p = []
        for k, l in self.d.items():
            if len(l) > 0:
                s = sum(l) / len(l) / args.grad_acc_steps
                p.append(f"{k}: {round(s, 6)}")
        return ", ".join(p)

    def clear(self):
        self.d["func"].clear()
        self.d["pos"].clear()
        self.d["neg"].clear()

    def __getitem__(self, k):
        return self.d[k]


def get_logits_from_lm(lm, inputs):
    outputs = lm(inputs)
    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(
        0
    )


def token_weighted_loss(loss_type, logits, inputs, weights):
    if loss_type == "ce":
        logits = logits.view(-1, logits.size(-1))
        inputs = inputs.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, inputs)
    elif loss_type == "ul":
        probs = F.softmax(logits, dim=-1)
        probs = torch.gather(probs, 2, inputs.unsqueeze(-1)).squeeze(-1)
        probs = torch.clamp((1.0 - probs), min=1e-5)
        loss = -torch.log(probs)
    else:
        assert False

    loss = loss[weights != 0]
    return loss.mean()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def step(self, batch):
        loss_dict = LossDict()

        sample_types, inputs, weights = batch
        inputs = inputs.to(self.model.device)
        shift_inputs = inputs[..., 1:]
        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:]
        outputs = self.model(inputs)
        shift_logits = outputs.logits[..., :-1, :]

        loss_total = 0.0
        for sample_type in sample_types:
            if sample_type == FUNC:
                loss = token_weighted_loss("ce", shift_logits, shift_inputs, shift_weights)
                loss_dict["func"].append(loss.item())
                loss_total += loss
            elif sample_type == GOOD:
                loss = token_weighted_loss("ce", shift_logits, shift_inputs, shift_weights)
                loss_dict["pos"].append(loss.item())
                loss_total += loss
            elif sample_type == BAD:
                loss = token_weighted_loss("ul", shift_logits, shift_inputs, shift_weights)
                loss_dict["neg"].append(loss.item())
                loss_total += loss
            else:
                assert False

        return loss_total, loss_dict

    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = LossDict()
        for batch in val_dataloader:
            loss, loss_dict = self.step(batch)
            acc_loss_dict.step(loss_dict)
        return acc_loss_dict.pretty_print(self.args)

    def load_model(self):
        self.tokenizer, self.model = load_model(self.args.pretrain_name, self.args)
        self.model.train()

    def load_dataset(self):
        self.dataset = CodeDataset(self.args, self.tokenizer, "train")
        self.val_dataset = CodeDataset(self.args, self.tokenizer, "val")

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def run(self):
        self.load_model()
        self.load_dataset()

        self.args.logger.info(f"Training args {self.args}")

        batch_size = self.args.batch_size
        train_sampler = RandomSampler(self.dataset)
        train_dataloader = DataLoader(
            self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True
        )

        total_samples = len(self.dataset)
        batch_size = batch_size * self.args.grad_acc_steps
        total_steps = total_samples // batch_size * self.args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (not any(nd in n for nd in no_decay)) and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=total_steps
        )
        num_params = sum(p.numel() for p in self.model.parameters())

        self.args.logger.info("***** Running training *****")
        self.args.logger.info("  Num samples = %d", total_samples)
        self.args.logger.info("  Num epoch = %d", self.args.num_train_epochs)
        self.args.logger.info("  Batch size= 1")
        self.args.logger.info("  Total batch size (w. accumulation) = %d", batch_size)
        self.args.logger.info("  Gradient Accumulation steps = %d", self.args.grad_acc_steps)
        self.args.logger.info("  Total optimization steps = %d", total_steps)
        self.args.logger.info("  Num val samples = %d", len(self.val_dataset))
        self.args.logger.info("  Num parameters = %d", num_params)

        global_step, acc_loss_dict = 0, LossDict()
        set_seed(self.args.seed)
        self.model.train()
        for idx in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                loss, loss_dict = self.step(batch)
                loss /= self.args.grad_acc_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                acc_loss_dict.step(loss_dict)

                if (step + 1) % self.args.grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        acc_loss_pp = acc_loss_dict.pretty_print(self.args)
                        self.args.logger.info(
                            "epochs: %s/%d, steps: %s/%d, %s",
                            idx + 1,
                            self.args.num_train_epochs,
                            global_step,
                            total_steps,
                            acc_loss_pp,
                        )
                        acc_loss_dict.clear()

            if self.args.save_epochs > 0 and (idx + 1) % self.args.save_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_loss_pp = self.do_eval()
                self.model.train()
                self.args.logger.info("val epoch %s: %s", idx + 1, eval_loss_pp)
                output_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{idx+1}")
                last_output_dir = os.path.join(self.args.output_dir, f"checkpoint-last")
                self.args.logger.info(
                    "Saving model checkpoint to %s and %s", output_dir, last_output_dir
                )
                self.save(output_dir)
                self.save(last_output_dir)

        if (idx + 1) % self.args.save_epochs != 0:
            self.model.eval()
            with torch.no_grad():
                eval_loss_pp = self.do_eval()
            self.args.logger.info("final eval loss: %s", eval_loss_pp)
            # output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
            last_output_dir = os.path.join(self.args.output_dir, f"checkpoint-last")
            # self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
            self.args.logger.info("Saving model checkpoint to %s", last_output_dir)
            # self.save(output_dir)
            self.save(last_output_dir)
