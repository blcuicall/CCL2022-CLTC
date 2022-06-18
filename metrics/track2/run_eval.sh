SRC_PATH=./demo/src.txt
HYP_PATH=./demo/hyp.txt
REF_PATH=./demo/ref.txt
OUTPUT_PATH=./demo/output.txt

python pair2edits_char.py $SRC_PATH $HYP_PATH > $OUTPUT_PATH

perl evaluation.pl $OUTPUT_PATH ./demo/report.txt $REF_PATH
