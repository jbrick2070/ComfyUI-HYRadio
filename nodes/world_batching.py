import torch

class HYWorld_BatchCLIPTextEncode:
    """
    Loops over a list of text prompts and encodes them into a properly concatenated Batch CONDITIONING object. 
    This enables native JSON batching for the KSampler without external API scripting.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text_list": ("STRING_LIST",)
            }
        }
        
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode_batch"
    CATEGORY = "HYWorld/Batching"
    
    def encode_batch(self, clip, text_list):
        conds = []
        for text in text_list:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conds.append([[cond, {"pooled_output": pooled}]])
            
        # ComfyUI natively batches conditions if we just pass a list of standard conditionings.
        # However, to be a true batch Tensor, they usually need to be concatenated. 
        # For simplicity, returning a standard list of conds or a concatenated batch depending on nodes downstream.
        
        # SD3/Flux batching requires padding the sequence dimension (dim 1) to the max length
        out_cond = []
        out_pooled = []
        for c in conds:
            out_cond.append(c[0][0])
            out_pooled.append(c[0][1]["pooled_output"])
            
        import torch.nn.functional as F
        max_len = max([c.shape[1] for c in out_cond]) if out_cond else 0
        
        padded_conds = []
        for c in out_cond:
            pad_amt = max_len - c.shape[1]
            if pad_amt > 0:
                padded_conds.append(F.pad(c, (0, 0, 0, pad_amt)))
            else:
                padded_conds.append(c)
                
        final_cond = torch.cat(padded_conds, dim=0) if padded_conds else torch.empty(0)
        final_pooled = torch.cat(out_pooled, dim=0) if out_pooled else torch.empty(0)
        
        return ([[final_cond, {"pooled_output": final_pooled}]],)

NODE_CLASS_MAPPINGS = {
    "HYWorld_BatchCLIPTextEncode": HYWorld_BatchCLIPTextEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorld_BatchCLIPTextEncode": "2. HYWorld Batch CLIP Encode (List -> Batch)"
}
