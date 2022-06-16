import json
import argparse


def read_file(data_path, flag):
    example = []
    with open(data_path, "r", encoding="UTF-8") as fin:
        for line in fin:
            line = line.strip()
            data = json.loads(line)
            if (flag == "test"):
                f05 = list()
                for hyp in data["hyps"]:
                    f05.append(hyp["idx"])
                example.append(f05)
            elif flag == "score":
                example.append(data)
        return example


def write_to_file(args):
    # num is to record the numbers of hyps for any src
    num = read_file(args.test_path, "test")
    score = read_file(args.test_score_path, "score")
    with open(args.output_path, "w")  as fout:
        count = 0
        for i in range(len(num)):
            for j in range(len(num[i])):
                fout.write(str(i) + '\t' + str(j) + '\t' + str(score[count + j]) + "\n")
            count = count + len(num[i])

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument('--test_path', required=True, help='test path')
        parser.add_argument('--test_score_path', required=True,
                            help='the path of the generate score for the test dataset')
        parser.add_argument('--output_path', required=True,help='the path of the final file for submiting')
        args = parser.parse_args()

        write_to_file(args)
