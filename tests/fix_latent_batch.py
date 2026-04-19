import json

with open('workflows/HYRadio-Visual-Test.json', 'r', encoding='utf-8') as f:
    wf = json.load(f)

for n in wf['nodes']:
    if n['type'] == 'EmptyLatentImage':
        if len(n.get('widgets_values', [])) >= 3:
            n['widgets_values'][2] = 3  # Set batch size to 3

with open('workflows/HYRadio-Visual-Test.json', 'w', encoding='utf-8') as f:
    json.dump(wf, f, indent=2)

print('EmptyLatentImage batch size correctly aligned to 3 objects.')
