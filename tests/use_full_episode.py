import json

with open('workflows/HYRadio-Visual-Test.json', 'r', encoding='utf-8') as f:
    wf = json.load(f)

# The full 6-scene layout covering an 8 minute episode
full_scenes = [
    "Scene 1: Gaslight Mutation, Victorian streets shrouded in eerie glowing green mist, cobbled roads, dramatic.",
    "Scene 2: Abandoned greenhouse overgrown with massive glowing fungal fauna.",
    "Scene 3: Deep underground, dark stone tunnels illuminated by flickering lantern light.",
    "Scene 4: A cavernous industrial factory room filled with towering steampunk machinery, rust and shadow.",
    "Scene 5: The scientist's chaotic laboratory, scattered blueprints and bubbling vats, focused light.",
    "Scene 6: The city skyline breaking through the dense clouds of pollution, early morning dawn."
]
scene_json = json.dumps(full_scenes)

for n in wf['nodes']:
    if n['type'] == 'LoadAudio' or n['type'] == 'VHS_LoadAudio':
        # Point to the FULL 8 minute file
        if 'widgets_values' in n and len(n['widgets_values']) > 0:
            n['widgets_values'][0] = 'old_time_radio/test_real_audio.wav'
            
    if n['type'] == 'HYWorld_EnvironmentPromptBuilder':
        # Insert the full episode array into the Prompt Builder
        if 'widgets_values' in n and len(n['widgets_values']) > 0:
            n['widgets_values'][0] = scene_json

with open('workflows/HYRadio-Visual-Test.json', 'w', encoding='utf-8') as f:
    json.dump(wf, f, indent=2)

print('Updated JSON for Full Episode testing.')
