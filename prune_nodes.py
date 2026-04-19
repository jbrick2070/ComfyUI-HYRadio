import json

files = ['workflows/HYRadio-Visual-Test.json', 'workflows/HYRadio-Full-Pipeline.json']
for f in files:
    try:
        d = json.load(open(f))
        d['nodes'] = [n for n in d['nodes'] if n['id'] not in [1, 2, 3, 4, 5]]
        valid_node_ids = set([n['id'] for n in d['nodes']])
        new_links = []
        if 'links' in d:
            for link in d['links']:
                if link[1] in valid_node_ids and link[3] in valid_node_ids:
                    new_links.append(link)
            d['links'] = new_links
        json.dump(d, open(f, 'w'), indent=2)
        print(f'Pruned {f}')
    except Exception as e:
        print(e)
