import argparse
import os

from safecoder.trainer import Trainer
from safecoder.utils import set_logging, set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", required=True)

    parser.add_argument("--pretrain_name", type=str, default="codegen-350m")
    parser.add_argument("--vul_type", type=str, default=None)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_num_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc_steps", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--data_dir", type=str, default="../data_train_val")
    parser.add_argument("--model_dir", type=str, default="../trained/")

    args = parser.parse_args()

    args.output_dir = os.path.join(args.model_dir, args.output_name)

    return args


def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, "train.log"))
    set_seed(args.seed)
    Trainer(args).run()


if __name__ == "__main__":
    main()
