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

    def gradient(self, pts, create_graph=True):
        return self.sdf_network.gradient(pts, create_graph=create_graph)
    def color(self, pts, gradients, dirs, feature_vector):
        return self.color_network(pts, gradients, dirs, feature_vector)

    def forward(self, pts, dirs, create_graph=True):
        gradients, sdf_nn_output = self.sdf_network.gradient(pts, create_graph=create_graph, return_output=True)
        gradients = gradients.squeeze()
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        color = self.color(pts, gradients, dirs, feature_vector)
        return {
            'sdf': sdf,
            'color': color,
            'feature_vector': feature_vector,
            'gradients': gradients
        }
    def get_params(self, lr):
        params = [
            {'params': self.nerf_outside.parameters(), 'lr': lr},
            {'params': self.sdf_network.parameters(), 'lr': lr},
            {'params': self.deviation_network.parameters(), 'lr': lr},
            {'params': self.color_network.parameters(), 'lr': lr},
        ]
        return params