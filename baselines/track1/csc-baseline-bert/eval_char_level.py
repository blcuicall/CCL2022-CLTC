# -*- coding:UTF-8 -*-
# @Author: Xuezhi Fang, Tianxin Liao
# @Date: 2022-5-9
# @Email: jasonfang3900@gmail.com, 329932916@qq.com

from collections import OrderedDict
import argparse


def read_data(path):
        with open(path, "r", encoding="utf8") as fr:
                for line in fr:
                        line = line.strip()
                        if not line or line.startswith("#"):
                                continue
                        yield line


def get_correction_map(path):
        all_map = OrderedDict()
        for line in read_data(path):
                items = line.split(",")
                item_id = items[0].strip()
                if len(items) % 2 != 0:
                        cor_map = dict()
                        for idx in range(1, len(items), 2):
                                pos = items[idx].strip()
                                cor_char = items[idx+1].strip()
                                cor_map[pos] = cor_char
                        all_map[item_id] = cor_map
                else:
                        all_map[item_id] = dict()
        return all_map


def get_eval_metrics(conf_set, total):
        # print(conf_set)
        # fp = len(conf_set["fp"]) / (len(conf_set["fp"]) + len(conf_set["tn"]))
        acc = (len(conf_set["tp"]) + len(conf_set["tn"])) / total
        prec = len(conf_set["tp"]) / (len(conf_set["tp"]) + len(conf_set["fp"]))
        recall = len(conf_set["tp"]) / (len(conf_set["tp"]) + len(conf_set["fn"]))
        f1 = 2.0 * prec * recall / (prec + recall)
        metrics = OrderedDict({"Accuracy": acc * 100, "Precision": prec * 100, "Recall": recall * 100, "F1": f1 * 100})
        return metrics


def get_char_metrics(hyp_path, gold_path):
        gold_map = get_correction_map(gold_path)
        hyp_map = get_correction_map(hyp_path)
        detect_conf_dict = {k: set() for k in ["tp", "fn", "fp", "tn"]}
        correct_conf_dict = {k: set() for k in ["tp", "fn", "fp", "tn"]}
        for item_id in gold_map:
                # gold: correct
                # print(gold_map[item_id])
                if not gold_map[item_id]:
                        if not hyp_map[item_id]:
                                detect_conf_dict["tn"].add(item_id)
                                correct_conf_dict["tn"].add(item_id)
                        else:
                                detect_conf_dict["fp"].add(item_id)
                                correct_conf_dict["fp"].add(item_id)
                        continue
                # gold: with error, pred: correct
                if not hyp_map[item_id]:
                        detect_conf_dict["fn"].add(item_id)
                        correct_conf_dict["fn"].add(item_id)
                        continue
                if gold_map[item_id].keys() == hyp_map[item_id].keys():
                        if set(gold_map[item_id].values()) == set(hyp_map[item_id].values()):
                                correct_conf_dict["tp"].add(item_id)
                        else:
                                correct_conf_dict["fn"].add(item_id)
                        detect_conf_dict["tp"].add(item_id)
                else:
                        detect_conf_dict["fn"].add(item_id)
                        correct_conf_dict["fn"].add(item_id)
        detect_metrics = get_eval_metrics(detect_conf_dict, len(hyp_map))
        correct_metrics = get_eval_metrics(correct_conf_dict, len(hyp_map))
        metrics = OrderedDict({"Detection": detect_metrics, "Correction": correct_metrics})
        print("=" * 10 + " Character Level " + "=" * 10)
        for k, v in metrics.items():
            print(f"{k}: ")
            print(", ".join([f"{k_i}: {round(v_i, 2)}" for k_i, v_i in v.items()]))
        return metrics

