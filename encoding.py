import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqEncoder_torch(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, freq_threshold=1.0, **kwargs):
        assert freq_threshold >= 0.0 and freq_threshold <= 1.0, 'progress in FreqEncoder should be in [0, 1]'

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            cur_freq_progress = i / len(self.freq_bands)

            # Coefficient computed using freq_threshold
            enc_coef = 1.0
            if cur_freq_progress > freq_threshold:
                enc_coef = 0.0
            elif cur_freq_progress > freq_threshold - 1 / len(self.freq_bands):
                enc_coef = (freq_threshold - cur_freq_progress) * len(self.freq_bands)
            
            for p_fn in self.periodic_fns:
                out_enc = p_fn(input * freq)
                out.append(out_enc * enc_coef)

        out = torch.cat(out, dim=-1)

        return out

class IncludeInputsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.input_dim = encoder.input_dim
        self.output_dim = encoder.output_dim + self.input_dim

    def forward(self, x, **kwargs):
        out = self.encoder(x, **kwargs)
        out = torch.cat([x, out], dim=-1)
        return out

def get_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False, interpolation='linear',
                include_inputs=False,
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim
    
    elif encoding == 'frequency_torch':
        encoder = FreqEncoder_torch(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)

    elif encoding == 'frequency': # CUDA implementation, faster than torch.
        from freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'sphere_harmonics':
        from shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners, interpolation=interpolation)
    
    elif encoding == 'tiledgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners, interpolation=interpolation)
    
    elif encoding == 'hashgrid_taichi':
        from taichi_modules.hash_encoder import HashEncoderTaichi
        encoder = HashEncoderTaichi(batch_size=4096) #TODO: hard encoded batch size

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    if include_inputs:
        encoder = IncludeInputsWrapper(encoder)

    return encoder, encoder.output_dim