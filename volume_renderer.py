import torch
from nerf.utils import custom_meshgrid, safe_normalize

@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', min_near=0.05):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=min_near)

    return near, far

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals
    assert len(bins.shape) == 2, 'bins should have shape [B, T]'
    B, T = bins.shape
    assert weights.shape == (B, T - 1), 'weights should have shape [B, T - 1]'

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

class VolumeRenderer(torch.nn.Module):
    def __init__(
            self,
            bound,
            min_near,
            num_steps,
            upsample_steps,
        ):
        super().__init__()
        self.bound = bound
        self.min_near = min_near
        self.num_steps = num_steps
        self.upsample_steps = upsample_steps
        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # Support for cuda / taichi off by default
        self.cuda_ray = False
        self.taichi_ray = False
    def run_core_weight_only(self, rays_o, rays_d, z_vals, sample_dist):
        raise NotImplementedError()
    def run_core(
            self, 
            rays_o,
            rays_d, 
            z_vals, 
            dirs, 
            light_d, 
            sample_dist,
            ambient_ratio=1.0,
            shading='albedo',
            bg_color=None,
            prefix=None
        ):
        raise NotImplementedError()
    def run(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        # nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        # nears.unsqueeze_(-1)
        # fars.unsqueeze_(-1)
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='sphere', min_near=self.min_near)
        sample_dist = (fars - nears) / self.num_steps

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, self.num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, self.num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / self.num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        # upsample z_vals (nerf-like)
        if self.upsample_steps > 0:
            with torch.no_grad():

                weights = self.run_core_weight_only(rays_o, rays_d, z_vals, sample_dist)
                if weights.shape[-1] == z_vals.shape[-1]:
                    # Weights corresponding to each point
                    # So we need to sample intermediate points
                    deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                    deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                    # sample new z_vals
                    z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                    new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], self.upsample_steps, det=not self.training).detach() # [N, t]
                elif weights.shape[-1] == z_vals.shape[-1] - 1:
                    # Weights already correspond to intermediate points
                    new_z_vals = sample_pdf(z_vals, weights, self.upsample_steps, det=not self.training).detach() # [N, T]
                else:
                    raise ValueError(f'weights.shape[-1] = {weights.shape[-1]} is not supported.')

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, _ = torch.sort(z_vals, dim=1)

        dirs = rays_d.view(-1, 1, 3).expand(-1, z_vals.shape[-1], -1)
        results = self.run_core(
            rays_o, 
            rays_d, 
            z_vals, 
            dirs, 
            sample_dist=sample_dist, 
            light_d=light_d, 
            ambient_ratio=ambient_ratio, 
            shading=shading,
            bg_color=bg_color,
            prefix=prefix
        )
        return results
    def render(self, rays_o, rays_d, mvp, h, w, staged=False, max_ray_batch=4096, **kwargs):
        return self.run(rays_o, rays_d, **kwargs)