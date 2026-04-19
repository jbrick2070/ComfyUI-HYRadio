import json
with open('workflows/HYRadio-Visual-Test.json', 'r', encoding='utf-8') as f:
    wf = json.load(f)

for n in wf['nodes']:
    if n['id'] == 102:
        # prompt
        n['widgets_values'][0] = '["Scene 1: Gaslight Mutation, Victorian streets shrouded in eerie glowing green mist, cobbled roads, dramatic."]'
    elif n['id'] == 201:
        # cinematic director
        n['widgets_values'][0] = '["{\\"preset\\\": \\"forward\\\", \\"fov_deg\\\": 70.0}"]'
    elif n['type'] == 'VHS_LoadAudio':
        # the audio node
        n['widgets_values'][0] = 'old_time_radio/test_visual_scene_1.wav'

with open('workflows/HYRadio-Visual-Test.json', 'w', encoding='utf-8') as f:
    json.dump(wf, f, indent=2)

print('Updated JSON to single-scene Real Audio mock.')
