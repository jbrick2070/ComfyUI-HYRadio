import json

def patch_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            wf = json.load(f)
        for n in wf.get('nodes', []):
            if 'flags' not in n: n['flags'] = {}
            if 'order' not in n: n['order'] = 0
            if 'mode' not in n: n['mode'] = 0
            if 'properties' not in n:
                n['properties'] = {"Node name for S&R": n['type']}
            if 'size' not in n:
                n['size'] = [315, 130]
            elif isinstance(n['size'], dict):
                # some sizes are dicts {"0": 315, "1": 130}, convert to list to be safe
                n['size'] = [n['size'].get("0", 315), n['size'].get("1", 130)]
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(wf, f, indent=2)
        print(f"Patched {path}")
    except Exception as e:
        print(f"Error on {path}: {e}")

patch_file('workflows/HYRadio-Visual-Test.json')
patch_file('workflows/HYRadio-Full-Pipeline.json')
