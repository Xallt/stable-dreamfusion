import unittest
import torch
import argparse

class TestRendererMethods(unittest.TestCase):
    def test_nerf_renderer(self):
        from nerf.network import NeRFNetwork
        from main import parse_args

        opt = parse_args("--text \"hamburger\" --workspace trial -O".split())

        nerf_network = NeRFNetwork(opt)
        nerf_network.training = False # Make everything deterministic

        rays_o = torch.rand(1, 10, 3)
        rays_d = torch.rand(1, 10, 3)

        result = nerf_network.run(rays_o, rays_d, perturb=False)

        keys = ["image", "depth", "weights", "weights_sum"]

        for key in keys:
            self.assertIn(key, result)
            value = result[key]
            self.assertIsInstance(value, torch.Tensor)
            self.assertFalse(torch.isnan(value).any())

    def test_neus_renderer(self):
        from pyhocon import ConfigFactory
        from neus.fields import SDFNetwork, RenderingNetwork, NeRF, SingleVarianceNetwork
        from neus.renderer import NeuSRenderer
        parser = argparse.ArgumentParser()
        parser.add_argument('--conf', type=str, default='./confs/base.conf')
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--mcube_threshold', type=float, default=0.0)
        parser.add_argument('--is_continue', default=False, action="store_true")
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--case', type=str, default='')

        args = parser.parse_args([
            '--conf', '/home/dmitry/clones/NeuS/confs/womask.conf',
            '--mode', 'train',
            '--mcube_threshold', '0.0',
            '--is_continue',
            '--gpu', '0',
            '--case', 'womask'
        ])

        f = open(args.conf)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', args.case)
        f.close()

        conf = ConfigFactory.parse_string(conf_text)

        device = 'cuda:0'

        renderer = NeuSRenderer(
            conf['model.nerf'],
            conf['model.sdf_network'],
            conf['model.variance_network'],
            conf['model.rendering_network'],
            **conf['model.neus_renderer']
        ).to(device)

        rays_o = torch.rand(1, 10, 3).to(device)
        rays_d = torch.rand(1, 10, 3).to(device)

        result = renderer.run(rays_o, rays_d)

        keys = ["image", "depth", "weights", "weights_sum"]

        for key in keys:
            self.assertIn(key, result)
            value = result[key]
            self.assertIsInstance(value, torch.Tensor)
            self.assertFalse(torch.isnan(value).any())