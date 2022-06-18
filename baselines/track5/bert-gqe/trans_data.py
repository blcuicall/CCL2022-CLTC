from base64 import encode
import json

if __name__ == "__main__":
    with open('./train/train.para', 'r', encoding='UTF-8') as fin:
        data_list = []
        for line in fin:
            line = line.strip()
            print(line)
            data1 = line.split('\t')[0]
            print(data1)
            data2 = line.split('\t')[1]
            print(data2)
            if data1 != data2:
                data_list.append(data1+'\t0')
                data_list.append(data2+'\t1')



with open('./train/train.json', 'w', encoding='utf-8') as fout:
    for index in data_list:
        fout.write(index + '\n')
