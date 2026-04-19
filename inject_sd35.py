import json

def inject_sd35():
    file_path = 'workflows/HYRadio-World-Mirror-panorama.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        wf = json.load(f)

    # Remove Node 17
    wf['nodes'] = [n for n in wf['nodes'] if n['id'] != 17]
    
    # Base configuration for SD 3.5 Large
    new_nodes = [
        {
            "id": 100, "type": "CheckpointLoaderSimple", "pos": [50, 50],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1000]}, 
                        {"name": "CLIP", "type": "CLIP", "links": [1001, 1002]}, 
                        {"name": "VAE", "type": "VAE", "links": [1003]}],
            "widgets_values": ["sd3.5_large_fp8_scaled.safetensors"]
        },
        {
            "id": 101, "type": "EmptyLatentImage", "pos": [50, 200],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [1004]}],
            "widgets_values": [2048, 1024, 1]
        },
        {
            "id": 102, "type": "CLIPTextEncode", "pos": [400, 50],
            "inputs": [{"name": "clip", "type": "CLIP", "link": 1001}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [1005]}],
            "widgets_values": ["A sweeping panoramic landscape of alien dunes, highly detailed"]
        },
        {
            "id": 103, "type": "CLIPTextEncode", "pos": [400, 200],
            "inputs": [{"name": "clip", "type": "CLIP", "link": 1002}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [1006]}],
            "widgets_values": ["watermark, text, borders, bad quality"]
        },
        {
            "id": 104, "type": "KSampler", "pos": [800, 50],
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 1000},
                {"name": "positive", "type": "CONDITIONING", "link": 1005},
                {"name": "negative", "type": "CONDITIONING", "link": 1006},
                {"name": "latent_image", "type": "LATENT", "link": 1004}
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [1007]}],
            "widgets_values": [123456789, "randomize", 30, 4.5, "dpmpp_2m", "karras", 1.0]
        },
        {
            "id": 105, "type": "VAEDecode", "pos": [1200, 50],
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": 1007},
                {"name": "vae", "type": "VAE", "link": 1003}
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [19, 20]}]
        }
    ]
    
    # New Links map: [link_id, from_node, from_slot, to_node, to_slot, type]
    new_links = [
        [1000, 100, 0, 104, 0, "MODEL"],
        [1001, 100, 1, 102, 0, "CLIP"],
        [1002, 100, 1, 103, 0, "CLIP"],
        [1003, 100, 2, 105, 1, "VAE"],
        [1004, 101, 0, 104, 3, "LATENT"],
        [1005, 102, 0, 104, 1, "CONDITIONING"],
        [1006, 103, 0, 104, 2, "CONDITIONING"],
        [1007, 104, 0, 105, 0, "LATENT"]
    ]
    
    wf['nodes'].extend(new_nodes)
    wf['links'].extend(new_links)
    
    # Update links 19 and 20 to point to VAEDecode (node 105) instead of LoadImage (node 17)
    for link in wf['links']:
        if type(link) == list and link[0] in [19, 20]:
            link[1] = 105  # new from_node
    
    wf['last_node_id'] = max(wf.get('last_node_id', 0), 105)
    wf['last_link_id'] = max(wf.get('last_link_id', 0), 1007)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(wf, f, indent=2)

if __name__ == "__main__":
    inject_sd35()
