import json
import random
import numpy as np

base_results = json.load(open("bert_base_analysis.json"))
kdn_results = json.load(open("kdn_analysis.json"))

assert len(base_results) == len(kdn_results)

analysis = []
for k in base_results.keys():
    base_num = base_results[k]
    kdn_num = kdn_results[k]
    if kdn_num["f1"] > base_num["f1"] and int(kdn_num["em"]) == 1:
        analysis.append((base_num, kdn_num))

random.shuffle(analysis)

for idx, a in enumerate(analysis[:100]):
    print(f"index: {idx}")
    print(a[0]["q"])
    print(a[0]['c'])
    print(a[0]['gold'])
    print(f"base: {a[0]['pred']}")
    print(f"kdn: {a[1]['pred']}")
    import pdb;pdb.set_trace()


print(np.sum([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

"""
Wikipedia entity only in Questions: [1, 1, 1, 1, 1, 1, 1, 1, 1]

Answers are Wikipedia entities: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]


"""
