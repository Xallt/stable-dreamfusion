import torch
from .renderer import NeuSRenderer
from .fields import SDFNetwork, RenderingNetwork, SingleVarianceNetwork
from .embedder import get_embedder
from nerf.network import BackgroundNetwork


class NeuSNetwork(NeuSRenderer):
    def __init__(
            self,
            sdf_network_config,
            variance_network_config,
            rendering_network_config,
            background_network_config,
            n_samples,
            n_outside,
            upsample_steps,
            bg_radius,
            perturb,
        ):
        super().__init__(
            n_samples,
            n_outside,
            upsample_steps,
            bg_radius,
            perturb
        )

        self.sdf_network = SDFNetwork(**sdf_network_config)
        self.deviation_network = SingleVarianceNetwork(**variance_network_config)
        self.color_network = RenderingNetwork(**rendering_network_config)
        # Learnable background color
        self.background_network = BackgroundNetwork(**background_network_config)

    def sdf(self, pts):
        return self.sdf_network.sdf(pts)

    def deviation(self, pts):
        return self.deviation_network(pts)

    def gradient(self, pts, create_graph=True):
        return self.sdf_network.gradient(pts, create_graph=create_graph)
    def color(self, pts, gradients, dirs, feature_vector):
        return self.color_network(pts, gradients, dirs, feature_vector)

    def background_color(self, pts, dirs):
        return self.background_network(dirs)
    def set_warmup_progress(self, progress):
        self.sdf_network.set_warmup_progress(progress)

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
            {'params': self.sdf_network.parameters(), 'lr': lr},
            {'params': self.deviation_network.parameters(), 'lr': lr},
            {'params': self.color_network.parameters(), 'lr': lr},
            {'params': self.background_network.parameters(), 'lr': lr * 0.1},
        ]
        return params