python train.py --output_name sec-evol-1b --datasets sec evol-66k --pretrain_name starcoderbase-1b
python sec_eval.py --output_name evol-1b --model_name evol-1b
./human_eval.sh sec-evol-1b sec-evol-1b
python sec_eval.py --output_name sec-evol-1b --model_name sec-evol-1b
