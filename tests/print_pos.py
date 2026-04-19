import json
with open('workflows/HYRadio-Visual-Test.json', 'r', encoding='utf-8') as f:
    wf = json.load(f)
for n in wf['nodes']:
    print(f"{n['id']:>3}: {n['type']:<30} pos={n['pos']}")
