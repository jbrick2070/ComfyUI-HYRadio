import json
with open('workflows/HYRadio-Visual-Test.json', 'r', encoding='utf-8') as f:
    wf = json.load(f)

node_ids = {n['id'] for n in wf['nodes']}
print(f'Total nodes: {len(node_ids)}')

links = wf['links']
print(f'Total links: {len(links)}')

# 1. Structural checks
has_errors = False
for l in links:
    idx, src, src_slot, dst, dst_slot, dtype = l
    if src not in node_ids:
        print(f'ERROR: Link {idx} references missing source node {src}')
        has_errors = True
    if dst not in node_ids:
        print(f'ERROR: Link {idx} references missing destination node {dst}')
        has_errors = True

# 2. Slot collisions
dst_slots = set()
for l in links:
    idx, src, src_slot, dst, dst_slot, dtype = l
    k = (dst, dst_slot)
    if k in dst_slots:
        print(f'ERROR: Link {idx} overwrites existing slot on node {dst} (slot {dst_slot})')
        has_errors = True
    dst_slots.add(k)

# 3. Last IDs check
last_link_id = max([l[0] for l in links], default=0)
last_node_id = max(list(node_ids), default=0)

if last_link_id > wf.get("last_link_id", 0):
    print(f'Warning: last_link_id needs update. Max is {last_link_id}, file has {wf.get("last_link_id")}')
if last_node_id > wf.get("last_node_id", 0):
    print(f'Warning: last_node_id needs update. Max is {last_node_id}, file has {wf.get("last_node_id")}')

if not has_errors:
    print('Integrity check PASSED. Graph is structurally valid.')
else:
    print('Integrity check FAILED.')
