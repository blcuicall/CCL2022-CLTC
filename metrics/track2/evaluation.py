import sys
hyp_path = sys.argv[1]
ref_path = sys.argv[2]
with open(ref_path) as f:
    lines = f.readlines()
from operator import truth
import re
corr = dict()
erro_exsist = dict()
erro_ident = dict()
erro_pos = dict()
erro_cor = dict()

truth_count = 0
for l in lines:
    l = l.strip()
    rlt = re.search(r'^(\S+),\s+(correct)', l)
    if rlt:
        corr[rlt.group(1)] = 1
    else:
        rlt = re.search(r'^(\S+),\s+(\d+),\s+(\d+),\s+([RWMS])', l)
        if rlt:
            erro_exsist[rlt.group(1)] = 1
            erro_ident[rlt.group(1)+' '+rlt.group(4)] = 1
            erro_pos[' '.join([rlt.group(1), rlt.group(2), rlt.group(3), rlt.group(4)])] = 1
    truth_count += 1

    rlt = re.search(r'^(\S+),\s+(\d+),\s+(\d+),\s+([MS]),\s+(\S.*)', l)
    if rlt:
        id, st, en, ty, anses = rlt.group(1), rlt.group(2), rlt.group(3), rlt.group(4), rlt.group(5)
        if re.search(r',', anses):
            for k in anses.split(','):
                k = k.strip()
                erro_cor[' '.join([id, st, en, ty, k])] = 1
        else:
            erro_cor[' '.join([id, st, en, ty, anses])] = 1
corr_count = len(corr)
erro_exsist_count = len(erro_exsist)
erro_ident_count = len(erro_ident)
erro_pos_count = len(erro_pos)
erro_cor_count = len(erro_cor)

# print(f"Correct Units: {corr_count}\t Units With Errors: {erro_exsist_count}\nError Type Counts in All Units: {erro_ident_count}\tError Count: {erro_pos_count}\t Erros Need Correction: {erro_cor_count}")
# print("\n\n=====================\n\n")


with open(hyp_path) as f:
    lines = f.readlines()
from operator import truth
import re
hash = dict()
hash_corr = dict()
hash_exsist = dict()
hash_ident = dict()
hash_pos = dict()
hash_cor = dict()

truth_count = 0
for l in lines:
    l = l.strip()
    rlt = re.search(r'^(\S+),\s+(correct)', l)
    if rlt:
        hash[rlt.group(1)] = 'correct'
        hash_corr[rlt.group(1)] = 1
    else:
        rlt = re.search(r'^(\S+),\s+(\d+),\s+(\d+),\s+([RWMS])', l)
        if rlt:
            hash[rlt.group(1)] = 1
            hash_exsist[rlt.group(1)] = 1
            hash_ident[rlt.group(1)+' '+rlt.group(4)] = 1
            hash_pos[' '.join([rlt.group(1), rlt.group(2), rlt.group(3), rlt.group(4)])] = 1
    truth_count += 1

    rlt = re.search(r'^(\S+),\s+(\d+),\s+(\d+),\s+([MS]),\s+(\S.*)', l)
    if rlt:
        id, st, en, ty, anses = rlt.group(1), rlt.group(2), rlt.group(3), rlt.group(4), rlt.group(5)
        if re.search(r',', anses):
            for k in anses.split(','):
                
                k = k.strip()
                if not k:
                    continue
                hash_cor[' '.join([id, st, en, ty, k])] = 1
        else:
            hash_cor[' '.join([id, st, en, ty, anses])] = 1
hash_corr_count = len(hash_corr)
hash_exsist_count = len(hash_exsist)
hash_ident_count = len(hash_ident)
hash_pos_count = len(hash_pos)
hash_cor_count = len(hash_cor)

# print (f"System Correct Units: {hash_corr_count}\tSystem Units With Errors: {hash_exsist_count}\nSystem Error Type Counts in All Units: {hash_ident_count}\tSystem Error Count: {hash_pos_count}\tSystem Corrected Erros:{hash_cor_count}")
# print (f"\n\n=====================\n\n")

fp_exsist = 0
tn_exsist = 0
for k in corr.keys():
    if k in hash.keys():
        if hash[k] != 'correct':
            fp_exsist += 1
        else:
            tn_exsist += 1
    else:
        fp_exsist += 1
fpr = fp_exsist / corr_count
# print(f'False Positive Rate = {fpr} ( {fp_exsist} / {corr_count})')
# print("\n\n=====================\n\n")


#detection
tp_exsist = 0
fn_exsist = 0
for k in erro_exsist.keys():
    if k in hash.keys():
        if hash[k] != 'correct':
            tp_exsist += 1
        else:
            fn_exsist += 1
    else:
        fn_exsist += 1
pre_exsist = tp_exsist / hash_exsist_count
rec_exsist = tp_exsist / erro_exsist_count
f1_detection = 2 * pre_exsist * rec_exsist / (pre_exsist + rec_exsist)
# print (f"Detction Level\nPre = {pre_exsist} ({tp_exsist} / hash_exsist_count)\nRec = {rec_exsist} ({tp_exsist} / {erro_exsist_count})\nF1 = {f1_detection} (2* {pre_exsist} * {rec_exsist} /( {pre_exsist} + {rec_exsist} ))")	
# print ("\n\n=====================\n\n")

#identification
tp_ident = 0
fn_ident = 0
for k in erro_ident.keys():
    if k in hash_ident.keys():
        tp_ident += 1
    else:
        fn_ident += 1
pre_ident=tp_ident/hash_ident_count
rec_ident=tp_ident/erro_ident_count
f1_identification=2*pre_ident*rec_ident/(pre_ident+rec_ident)
# print (f"Identification Level\nPre = {pre_ident} ({tp_ident} / {hash_ident_count})\nRec = {rec_ident} ({tp_ident} / {erro_ident_count})\nF1 = {f1_identification} (2* {pre_ident} * {rec_ident} / ( {pre_ident}+{rec_ident} ))")	
# print ("\n\n=====================\n\n")

#position
tp_pos = 0
tn_pos = 0
for k in erro_pos.keys():
    if k in hash_pos.keys():
        tp_pos += 1
    else:
        tn_pos += 1

pre_pos=tp_pos/hash_pos_count
rec_pos=tp_pos/erro_pos_count
if(pre_pos+rec_pos>0):
	f1_position=2*pre_pos*rec_pos/(pre_pos+rec_pos)
else:
	f1_position=0
# print (f"Postion Level\nPre = {pre_pos} ( {tp_pos} / {hash_pos_count} )\nRec = {rec_pos} ( {tp_pos} / {erro_pos_count} )\nF1 = {f1_position} ( 2 * {pre_pos} * {rec_pos} /( {pre_pos}+{rec_pos} ) )")
# print ("\n\n=====================\n\n")

# 修正track
hit_top3 = 0
for k in erro_cor.keys():
    if k in hash_cor.keys():
        hit_top3 += 1

if(hash_cor_count>0):
	pre_top3=hit_top3/hash_cor_count
	rec_top3=hit_top3/erro_cor_count

if(pre_top3+rec_top3>0):
	f1_top3=2*pre_top3*rec_top3/(pre_top3+rec_top3)
else:
	f1_top3=0

# print (f"Correction Level\nPre = {pre_top3} ( {hit_top3} / {hash_cor_count} )\nRec = {rec_top3} ( {hit_top3} / {erro_cor_count} )\nF1 = {f1_top3} ( 2 * {pre_top3} * {rec_top3} /( {pre_top3} + {rec_top3} ) )\n")

# print ("\n\n==========END===========\n\n")

fpr, f1_detection, f1_identification, f1_position, f1_top3
comp_score = 0.25 * (f1_detection+f1_identification+f1_position+f1_top3-fpr)
result = {"FPR":fpr, "DET":f1_detection, "IDE":f1_identification, "POS":f1_position, "COR":f1_top3, "COM":comp_score}
import json
print(json.dumps(result))