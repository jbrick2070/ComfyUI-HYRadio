import json
import os

def sanitize_links(path):
    if not os.path.exists(path):
        print(f"Skipping {path}, file not found.")
        return
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            wf = json.load(f)
        
        links = wf.get('links', [])
        nodes = wf.get('nodes', [])
        
        # Build mapping for quick lookup: link_id -> link_data
        link_map = {l[0]: l for l in links}
        
        # Build mapping for output links: (node_id, slot_index) -> [link_ids]
        output_links = {}
        for l in links:
            lid, src_id, src_slot, dst_id, dst_slot, dtype = l
            key = (src_id, src_slot)
            if key not in output_links:
                output_links[key] = []
            output_links[key].append(lid)
            
        # Build mapping for input links: (node_id, slot_index) -> link_id
        input_links = {}
        for l in links:
            lid, src_id, src_slot, dst_id, dst_slot, dtype = l
            key = (dst_id, dst_slot)
            input_links[key] = lid

        # Sanitize nodes
        for n in nodes:
            nid = n['id']
            
            # Sanitize inputs
            if 'inputs' in n:
                for i, inp in enumerate(n['inputs']):
                    actual_lid = input_links.get((nid, i))
                    if actual_lid is not None:
                        inp['link'] = actual_lid
                    else:
                        # Ensure no invalid link IDs are left
                        if 'link' in inp:
                            # Check if the existing link ID actually points here
                            existing_lid = inp['link']
                            if existing_lid in link_map:
                                l = link_map[existing_lid]
                                if l[3] != nid or l[4] != i:
                                    inp['link'] = None
                            else:
                                inp['link'] = None
            
            # Sanitize outputs
            if 'outputs' in n:
                for j, out in enumerate(n['outputs']):
                    actual_lids = output_links.get((nid, j), [])
                    out['links'] = actual_lids

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(wf, f, indent=2)
        print(f"Successfully sanitized links in {path}")

    except Exception as e:
        print(f"Error sanitizing {path}: {e}")

# Run for both active workflows
sanitize_links('workflows/HYRadio-Visual-Test.json')
sanitize_links('workflows/HYRadio-Full-Pipeline.json')
