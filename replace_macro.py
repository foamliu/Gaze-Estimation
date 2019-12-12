# -*- coding: utf-8 -*-
import json

if __name__ == '__main__':
    with open('README.t', 'r', encoding="utf-8") as file:
        text = file.readlines()
    text = ''.join(text)

    with open('sample_preds.json', 'r', encoding="utf-8") as file:
        results = json.load(file)

    for i in range(10):
        item = results[i]
        print(item)
        result_true = ','.join(['{:.4f}'.format(i) for i in item['label']])
        result_out = ','.join(['{:.4f}'.format(i) for i in item['out']])
        text = text.replace('$(result_true_{})'.format(i), result_true)
        text = text.replace('$(result_out_{})'.format(i), result_out)

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(text)
