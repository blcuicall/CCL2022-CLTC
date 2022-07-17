import argparse
from collections import OrderedDict

def compute_prf(results):
    TP = 0
    FP = 0
    FN = 0
    wrong_char = 0
    all_predict_true_index = []
    all_gold_index = []
    for item in results:
        src, tgt, predict = item
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            wrong_char += 1
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)

    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0

    TP = 0
    FP = 0
    FN = 0

    for i in range(len( all_predict_true_index)):
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(results[i][2][j])
                if results[i][1][j] == results[i][2][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if results[i][1][j]  in predict_words:
                    continue
                else:
                    FN += 1

    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall) if (correction_precision + correction_recall) > 0 else 0

    metrics = OrderedDict()
    metrics["Detection"] = OrderedDict({
        'Precision': detection_precision * 100,
        'Recall': detection_recall * 100,
        'F1': detection_f1 * 100,
    })
    metrics["Correction"] = OrderedDict({
        'Precision': correction_precision * 100,
        'Recall': correction_recall * 100,
        'F1': correction_f1 * 100,
    })

    # print("=" * 10 + " Character Level " + "=" * 10)
    # print("Detection:")
    # print("Precision: {}, Recall: {}, F1: {}".format(round(detection_precision * 100, 2), round(detection_recall * 100, 2), round(detection_f1 * 100, 2)))
    # print("Detection:")
    # print("Precision: {}, Recall: {}, F1: {}".format(round(correction_precision* 100, 2), round(correction_recall * 100, 2), round(correction_f1 * 100, 2)))

    print("=" * 10 + " Character Level " + "=" * 10)
    for k, v in metrics.items():
        print(f"{k}: ")
        print(", ".join([f"{k_i}: {round(v_i, 2)}" for k_i, v_i in v.items()]))

    return metrics


def read_data(path, src):
    data = []
    with open(path, "r", encoding="utf8") as fin:
        lines = fin.readlines()
    for line, src_line in zip(lines, src):
        src_list = list(src_line)
        sent = src_line
        items = line.strip().split(", ")
        if len(items) == 2:
            pass
        else:
            for i in range(1, len(items), 2):
                src_list[int(items[i])-1] = items[i+1]
            sent = ''.join(src_list)
        data.append(sent)
    return data


def read_src(path):
    data = []
    with open(path, "r", encoding="utf8") as fin:
        lines = fin.readlines()
    for line in lines:
        items = line.strip().split("\t")
        data.append(items[1])
    return data


def main(config):
    src_path = config.src
    gold_path = config.gold
    pred_path = config.hyp
    src = read_src(src_path)
    pred_data = read_data(pred_path, src)
    gold_data = read_data(gold_path, src)
    result = []
    for i, j, k in zip(src, gold_data, pred_data):
        item = (i, j, k)
        result.append(item)
    return compute_prf(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--gold", type=str)
    parser.add_argument("--hyp", type=str)
    args = parser.parse_args()

    main(args)