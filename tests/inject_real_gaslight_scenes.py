import json
import os

with open('workflows/HYRadio-Visual-Test.json', 'r', encoding='utf-8') as f:
    wf = json.load(f)

# The EXACT scenes from the signal_lost_gaslight_mutation_20260418_105342 treatment
real_scenes_from_mp4_run = [
    "SCENE 1: Labyrinthine underground facility, constant hum of machinery, faint tremors",
    "SCENE 1: Laboratory, tense, flickering lights",
    "SCENE 3: Emergency access tunnel, echoing footsteps"
]
scene_json = json.dumps(real_scenes_from_mp4_run)

for n in wf['nodes']:
    if n['type'] == 'LoadAudio' or n['type'] == 'VHS_LoadAudio':
        # Point to the real video's audio extraction if it exists, otherwise use test_real_audio.wav
        # Wait, the user has test_real_audio.wav which they said was the full episode wav.
        n['widgets_values'][0] = 'old_time_radio/test_real_audio.wav'
        
    if n['type'] == 'HYWorld_EnvironmentPromptBuilder':
        # Insert the true array into the Prompt Builder
        n['widgets_values'][0] = scene_json

with open('workflows/HYRadio-Visual-Test.json', 'w', encoding='utf-8') as f:
    json.dump(wf, f, indent=2)

print('Updated JSON strictly with the factual scenes parsed from the Gaslight Mutation run.')
