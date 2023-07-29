import torch
from .renderer import NeuSRenderer
from .fields import SDFNetwork, RenderingNetwork, NeRF, SingleVarianceNetwork

class NeuSNetwork(NeuSRenderer):
    def __init__(
            self,
            nerf_outside_config,
            sdf_network_config,
            variance_network_config,
            rendering_network_config,
            n_samples,
            n_importance,
            n_outside,
            up_sample_steps,
            perturb
        ):
        super().__init__(
            n_samples,
            n_importance,
            n_outside,
            up_sample_steps,
            perturb
        )
        self.nerf_outside = NeRF(**nerf_outside_config)
        self.sdf_network = SDFNetwork(**sdf_network_config)
        self.deviation_network = SingleVarianceNetwork(**variance_network_config)
        self.color_network = RenderingNetwork(**rendering_network_config)

    def sdf(self, pts):
        return self.sdf_network.sdf(pts)

    def deviation(self, pts):
        return self.deviation_network(pts)

    def gradient(self, pts):
        return self.sdf_network.gradient(pts)
    def color(self, pts, gradients, dirs, feature_vector):
        return self.color_network(pts, gradients, dirs, feature_vector)

    def forward(self, pts, dirs):
        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        gradients = self.gradient(pts).squeeze()
        color = self.color(pts, gradients, dirs, feature_vector)
        return {
            'sdf': sdf,
            'color': color,
            'gradients': gradients
        }