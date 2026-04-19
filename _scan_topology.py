"""Bidirectional topology scan for ComfyUI workflow JSON."""
import json, sys

with open('workflows/HYRadio-Visual-Test.json', 'r') as f:
    wf = json.load(f)

links_by_id = {l[0]: l for l in wf.get('links', [])}
nodes_by_id = {n['id']: n for n in wf['nodes']}
issues = []

# 1. Every link_id in a node's outputs.links must exist in top-level links with matching origin
for n in wf['nodes']:
    nid = n['id']
    for si, outp in enumerate(n.get('outputs', [])):
        for lid in outp.get('links') or []:
            if lid not in links_by_id:
                issues.append(
                    f"ORPHAN OUTPUT: Node {nid} output[{si}] refs link {lid} -- NOT in links array"
                )
            else:
                link = links_by_id[lid]
                if link[1] != nid:
                    issues.append(
                        f"ORIGIN MISMATCH: Node {nid} output claims link {lid}, "
                        f"but links array says origin={link[1]}"
                    )
                if link[2] != si:
                    issues.append(
                        f"SLOT MISMATCH: Node {nid} output[{si}] claims link {lid}, "
                        f"but links array says from_slot={link[2]}"
                    )

# 2. Every link in links array must target a real input slot
for l in wf.get('links', []):
    lid, from_id, from_slot, to_id, to_slot, ltype = l
    if to_id not in nodes_by_id:
        issues.append(f"MISSING TARGET: Link {lid} targets node {to_id} which doesnt exist")
        continue
    target = nodes_by_id[to_id]
    inputs = target.get('inputs', [])
    if to_slot >= len(inputs):
        issues.append(
            f"SLOT OVERFLOW: Link {lid} targets node {to_id} slot {to_slot}, "
            f"but node only has {len(inputs)} inputs"
        )
    else:
        inp = inputs[to_slot]
        if inp.get('link') != lid:
            issues.append(
                f"LINK REF MISMATCH: Link {lid} targets node {to_id} input[{to_slot}], "
                f"but that input refs link={inp.get('link')}"
            )
        if inp.get('type') != ltype:
            issues.append(
                f"TYPE MISMATCH: Link {lid} type={ltype} but node {to_id} "
                f"input[{to_slot}] type={inp.get('type')}"
            )
    if from_id not in nodes_by_id:
        issues.append(f"MISSING SOURCE: Link {lid} from node {from_id} which doesnt exist")

# 3. S&R property vs type mismatch
for n in wf['nodes']:
    snr = n.get('properties', {}).get('Node name for S&R', '')
    ntype = n.get('type', '')
    if snr and snr != ntype:
        issues.append(f"S_AND_R MISMATCH: Node {n['id']} type={ntype} but S_R={snr}")

print(f"Total issues: {len(issues)}")
for iss in issues:
    print(iss)

if not issues:
    print("ALL CLEAR: No topology issues found.")
