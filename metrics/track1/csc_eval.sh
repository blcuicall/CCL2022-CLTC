#!/bin/bash
hyp=$1
gold=$2

echo -e "CSC Evaluation Report: \n"
python eval_sent_level.py --hyp $hyp --gold $gold
echo ""
python eval_char_level.py --hyp $hyp --gold $gold
