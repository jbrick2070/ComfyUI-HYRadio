import json

with open('workflows/HYRadio-Visual-Test.json', 'r', encoding='utf-8') as f:
    wf = json.load(f)

# Hardcoded layout definitions X, Y
layout = {
    100: [50, 50],
    101: [50, 300],
    2013: [50, 550],  # LoadAudio

    102: [400, 50],
    103: [400, 300],
    201: [400, 550],

    104: [800, 50],
    1: [800, 300],
    13: [800, 500],

    105: [1200, 50],
    2: [1200, 300],
    14: [1200, 500],

    107: [1600, 50],
    3: [1600, 300],
    15: [1600, 500],

    4: [2000, 300],
    12: [2000, 500],

    5: [2400, 300],
    16: [2400, 500],

    2012: [2800, 50]  # Final Video
}

# Assign any unmapped node to dynamically expanding locations below
missing_y = 800
missing_x = 50

for n in wf['nodes']:
    nid = n['id']
    if nid in layout:
        n['pos'] = layout[nid]
    else:
        n['pos'] = [missing_x, missing_y]
        missing_x += 400
        if missing_x > 2000:
            missing_x = 50
            missing_y += 200

with open('workflows/HYRadio-Visual-Test.json', 'w', encoding='utf-8') as f:
    json.dump(wf, f, indent=2)

print('Aesthetics and node positioning updated.')
