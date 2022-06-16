import json
import argparse
import numpy as np

def pearson_correlation(pred, ref):
    """ Computes Pearson correlation """
    from scipy.stats import pearsonr
    pc = pearsonr(pred, ref)
    return pc[0]

def read_file(data_path, flag):
    example = []
    with open(data_path, "r", encoding="UTF-8") as fin:

        # 获取真实f0.5分数
        if flag == "f05":
            for line in fin:
                f05 = list()
                line = line.strip()
                data = json.loads(line)
                for hyp in data["hyps"]:
                    f05.append(hyp["f05"])
                example.append(f05)

        # 获取生成的预测分数
        elif flag == "score":
            for line in fin:
                line = line.strip()
                data = line.split('\t')[2]
                example.append(data)

        # 获取最大的id 原句 假设句子
        elif flag == "hyp":
            src_id = 0  # 在获得的最大hyp_id中相应的原句id
            for line in fin:
                line = line.strip()
                data = json.loads(line)
                for hyp in data["hyps"]:
                    if hyp["idx"] == hyp_idx[src_id]:
                        example.append(str(src_id)+'\t'+data["src"]+'\t'+hyp["text"])
                src_id = src_id + 1
                
        else:
            raise Exception('flag error')

        return example

      
# get id src maxhyp ---> file
def maxhyp_to_file(args):
    hyps = read_file(args.test_f05_path, "hyp")

    # 将hyps写入文件
    f = open(args.hyp_file, "w", encoding='utf-8')
    f.write('\n'.join(hyps))
    f.close()

def sum_avarge_pcc(args):
    # get pcc & hyp_idx
    f05 = read_file(args.test_f05_path, "f05")
    score = read_file(args.baseline_results_path, "score")
    # 记录已有的hyp数量
    count = 0
    pcc = list()
    # hyp_idx:每个原句中分值最高的修改句id
    hyp_idx = list()
    for src_idx in range(len(f05)):
        # ------------------计算平均pcc ------------
        f = f05[src_idx]
        s = score[count:count + len(f)]
        if set(s) == {0}:
            pcc.append(0)
        else:
            pcc.append(pearson_correlation(f, s))
        hyp_maxscore = max(s)
        hyp_idx.append(s.index(hyp_maxscore))
        count = count + len(f)
    pcc = [i for i in pcc if i == i]
    pcc_tmp = np.array(pcc)
    pcc_tmp2 = np.mat(pcc_tmp)
    pcc_score = np.mean(pcc_tmp2)
    return hyp_idx, pcc_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--test_f05_path', required=True, help='the path for the test dataset with f0.5')
    parser.add_argument('-c','--hyp_file', required=True, help='the path of combined file ( id src hpy) for m2')
    parser.add_argument('-b','--baseline_results_path', required=True, help='the final file for submiting')
    args = parser.parse_args()

    hyp_idx,pcc_score=sum_avarge_pcc(args)

    print('pcc_score:',pcc_score)

    maxhyp_to_file(args)
    print("------将top-1的hyp写入文件用于后序评测------")
