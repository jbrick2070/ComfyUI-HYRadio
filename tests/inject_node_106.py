import json

with open('workflows/HYRadio-Visual-Test.json', 'r', encoding='utf-8') as f:
    wf = json.load(f)

# Re-inject Node 106 if it doesn't exist
if not any(n['id'] == 106 for n in wf['nodes']):
    node_106 = {
      'id': 106,
      'type': 'HYWorld_EnvironmentPromptBuilder',
      'pos': [50, 800],
      'size': [315, 200],
      'flags': {}, 'order': 0, 'mode': 0,
      'inputs': [],
      'outputs': [
        {
          'name': 'environmental_prompts',
          'type': 'STRING_LIST',
          'links': [6001]
        },
        {
          'name': 'cinematic_directives',
          'type': 'STRING_LIST',
          'links': [6002]
        }
      ],
      'properties': {'Node name for S&R': 'HYWorld_EnvironmentPromptBuilder'},
      'widgets_values': [
        '["Scene 1: Gaslight Mutation, Victorian streets shrouded in eerie glowing green mist, cobbled roads, dramatic."]',
        60
      ]
    }
    wf['nodes'].append(node_106)

    # Add missing links if they aren't there
    wf['links'].extend([
      [6001, 106, 0, 102, 0, 'STRING_LIST'],
      [6002, 106, 1, 201, 0, 'STRING_LIST']
    ])

    # Update 102 and 201 to accept these links
    for n in wf['nodes']:
        if n['id'] == 102:
            n['inputs'] = [{'name': 'text_list', 'type': 'STRING_LIST', 'link': 6001}]
            # Clear strings from widget we put earlier
            if len(n.get('widgets_values', [])) > 0:
                n['widgets_values'].pop(0)
                
        if n['id'] == 201:
            n['inputs'] = [{'name': 'cinematic_directives', 'type': 'STRING_LIST', 'link': 6002}]
            if len(n.get('widgets_values', [])) > 0:
                n['widgets_values'].pop(0)

    # Update last node and link ID
    wf['last_node_id'] = max(wf.get('last_node_id', 0), 106)
    wf['last_link_id'] = max(wf.get('last_link_id', 0), 6002)

    with open('workflows/HYRadio-Visual-Test.json', 'w', encoding='utf-8') as f:
        json.dump(wf, f, indent=2)

    print('Re-injected Node 106 Environment Prompt Builder.')
else:
    print('Node 106 already exists.')
