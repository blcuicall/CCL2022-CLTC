#!/bin/bash
hyp=$1
gold=$2

echo -e "CSC Evaluation Report: \n"
python eval_sent_level.py --hyp $hyp --gold $gold --json results/eval_sent_level.json
echo ""
python eval_char_level.py --hyp $hyp --gold $gold --json results/eval_char_level.json
