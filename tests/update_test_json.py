import json
with open('workflows/HYRadio-Visual-Test.json', 'r', encoding='utf-8') as f:
    wf = json.load(f)

# Delete node 106
wf['nodes'] = [n for n in wf['nodes'] if n['id'] != 106]
wf['links'] = [l for l in wf['links'] if l[1] != 106 and l[3] != 106]

for n in wf['nodes']:
    if n['id'] == 102:
        # Batch CLIP encode
        n['inputs'] = [inp for inp in n.get('inputs', []) if inp['name'] != 'text_list']
        if 'widgets_values' not in n:
            n['widgets_values'] = []
        n['widgets_values'].insert(0, '["Scene 1: Mars", "Scene 2: Moon"]')
    elif n['id'] == 201:
        # Cinematic Translator
        n['inputs'] = [inp for inp in n.get('inputs', []) if inp['name'] != 'cinematic_directives']
        if 'widgets_values' not in n:
            n['widgets_values'] = []
        n['widgets_values'].insert(0, '["{\\"preset\\\": \\"forward\\\", \\"fov_deg\\\": 70.0}", "{\\"preset\\\": \\"zoom_in\\\", \\"fov_deg\\\": 50.0}"]')

with open('workflows/HYRadio-Visual-Test.json', 'w', encoding='utf-8') as f:
    json.dump(wf, f, indent=2)

print('Updated JSON nodes to be standalone string fields')
