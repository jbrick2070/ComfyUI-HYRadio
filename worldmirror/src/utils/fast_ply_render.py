
import torch
import torch.nn.functional as F
import numpy as np
import time
import math

class FastPLYRenderer:
    """
    Super-fast PyTorch Point Cloud Renderer.
    Renders millions of points by projecting them and using scatter operations.
    No loops, no OpenGL, no custom CUDA.
    """
    
    def __init__(self, device):
        self.device = device

    def render(self, means, colors, opacities, scales, c2w, width, height, fov_deg=90.0, bg_color=(0.0, 0.0, 0.0)):
        """
        Render generic point cloud / splats via screen-space scattering.
        
        Args:
            means: [N, 3]
            colors: [N, 3] (0-1)
            opacities: [N] (0-1)
            scales: [N, 3] - used to determine point size
            c2w: [4, 4] camera-to-world
            width, height: resolution
        """
        # 1. Setup Camera
        fov_rad = math.radians(fov_deg)
        focal = width / (2 * math.tan(fov_rad / 2))
        cx, cy = width / 2, height / 2
        
        w2c = torch.linalg.inv(c2w).to(self.device).float()
        
        # 2. Project Points (World -> Camera)
        # Add homogeneous coord
        ones = torch.ones(means.shape[0], 1, device=self.device)
        points_h = torch.cat([means, ones], dim=1)
        
        # Transform
        points_cam = points_h @ w2c.T
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        
        # 3. Filter points behind camera
        valid_mask = z > 0.1
        if not valid_mask.any():
            return torch.zeros((height, width, 3), device=self.device)
            
        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        colors = colors[valid_mask]
        opacities = opacities[valid_mask]
        
        # Use max scale as approximate radius proxy
        if scales is not None:
             scales = scales[valid_mask].max(dim=1).values
        else:
             scales = torch.ones_like(z) * 0.01

        # 4. Project to Screen (Camera -> Image)
        # OpenCV convention: x right, y down. But Pytorch/OpenGL might differ.
        # We assume standard pinhole:
        # u = fx * x / z + cx
        # v = fy * y / z + cy
        # Note: If y is UP in camera space (OpenGL), we need to flip.
        # Our `_make_look_at_pose` uses OpenCV convention (Y down), so no flip needed?
        # Let's verify: In OpenCV, Y is down. +Y -> Bottom.
        
        u = (focal * x / z) + cx
        v = (focal * y / z) + cy
        
        # 5. Screen Bounds Check
        h_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        
        u = u[h_mask]
        v = v[h_mask]
        z = z[h_mask]
        colors = colors[h_mask]
        opacities = opacities[h_mask]
        scales = scales[h_mask]
        
        if u.numel() == 0:
            return torch.zeros((height, width, 3), device=self.device)

        # 6. Sort by Depth (Back-to-Front)
        # We want farthest points drawn first.
        # z is positive (distance). Large z = far.
        # argsort(z, descending=True) -> Far to Near.
        indices = torch.argsort(z, descending=True)
        
        u = u[indices]
        v = v[indices]
        colors = colors[indices]
        opacities = opacities[indices]
        scales = scales[indices]
        z = z[indices] # Need sorted z for point size calc
        
        # 7. Gaussian-kernel splat accumulator.
        # Instead of last-writer-wins 3x3 squares (which produced hard edges
        # and left sparse/pointillist gaps when cam was far from the scene),
        # each projected point contributes to a 5x5 Gaussian footprint with
        # depth-weighted alpha. Overlapping points blend via weighted average;
        # near points dominate over far ones via 1/(z^2+eps) depth weighting.
        # This is a cheap stand-in for gsplat's real Gaussian rasterization
        # — edges get soft falloff, gaps between projected points fill in
        # naturally, and the result looks less pointillist.

        # Build kernel once: 5x5 Gaussian, sigma=1, trimmed at w>=0.05.
        sigma = 1.0
        kernel = []
        for dv in range(-2, 3):
            for du in range(-2, 3):
                w = math.exp(-(du * du + dv * dv) / (2.0 * sigma * sigma))
                if w >= 0.05:
                    kernel.append((du, dv, float(w)))

        ui = u.long()
        vi = v.long()

        # Cast to fp32 for stable accumulation. V2 emits bf16; index_add_
        # requires matched dtypes across accumulators.
        if colors.dtype != torch.float32:
            colors = colors.float()
        if opacities.dtype != torch.float32:
            opacities = opacities.float()
        if z.dtype != torch.float32:
            z = z.float()

        # Depth-weighted alpha: near points dominate the weighted average,
        # approximating occlusion. Stronger than 1/z since z spans ~[0.1, 2]
        # for typical scenes; z^2 gives ~20x weight swing end-to-end.
        depth_weight = 1.0 / (z * z + 0.25)
        point_alpha = opacities * depth_weight  # [N]

        numer = torch.zeros(height * width, 3, device=self.device, dtype=torch.float32)
        denom = torch.zeros(height * width, device=self.device, dtype=torch.float32)

        for du, dv, kw in kernel:
            new_u = ui + du
            new_v = vi + dv
            in_frame = (new_u >= 0) & (new_u < width) & (new_v >= 0) & (new_v < height)
            if not in_frame.any():
                continue
            idx = new_v[in_frame] * width + new_u[in_frame]
            w_vec = kw * point_alpha[in_frame]             # [M]
            numer.index_add_(0, idx, colors[in_frame] * w_vec.unsqueeze(1))
            denom.index_add_(0, idx, w_vec)

        # Weighted average. Pixels never written stay black (denom=0 -> 0/eps).
        img = numer / (denom.unsqueeze(1) + 1e-6)
        img = img.reshape(height, width, 3)

        # Flip vertically (OpenCV -> screen top-left-origin mapping)
        img = torch.flip(img, [0])

        return img

