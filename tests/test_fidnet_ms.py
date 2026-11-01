"""Architecture-scope and end-to-end gradient checks; no dataset required."""

import unittest

import torch

from model.FIDNet import Amplitude_Phase_Block, FIDNet
from model.FIDNet_MS import FIDNetMS, MultiScaleAmplitudeConv


class FIDNetMSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_amplitude_branch_changes(self):
        torch.manual_seed(42)
        original = FIDNet(hid_S=8, hid_T=2)
        torch.manual_seed(42)
        modified = FIDNetMS(hid_S=8, hid_T=2)
        original_state = original.state_dict()
        modified_state = modified.state_dict()
        unchanged = {
            key for key in original_state if ".amp_fuse_1." not in key
        }
        self.assertEqual(
            unchanged,
            {key for key in modified_state if ".amp_fuse_1." not in key},
        )
        for key in unchanged:
            self.assertTrue(torch.equal(original_state[key], modified_state[key]), key)

        blocks = modified.fourierNet.enc_fre.enc
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertIsInstance(block, Amplitude_Phase_Block)
            self.assertIsInstance(block.amp_fuse_1, MultiScaleAmplitudeConv)
        # The original class remains usable as the baseline.
        for block in original.fourierNet.enc_fre.enc:
            self.assertNotIsInstance(block.amp_fuse_1, MultiScaleAmplitudeConv)

    def assert_valid_backward(self, model, x):
        model.zero_grad(set_to_none=True)
        y = model(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all().item())
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        amplitude_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if "amp_fuse_1" in name
        ]
        self.assertTrue(amplitude_parameters)
        for name, parameter in amplitude_parameters:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        return y.detach()

    def test_apb_preserves_even_and_odd_shapes(self):
        # APB itself supports odd sizes; the inherited 3D U-Net has stricter sizes.
        for height, width in ((16, 20), (17, 19)):
            with self.subTest(height=height, width=width):
                block = Amplitude_Phase_Block(8, 8, stride=1)
                block.amp_fuse_1 = MultiScaleAmplitudeConv(8)
                x = torch.randn(2, 8, height, width, requires_grad=True)
                self.assert_valid_backward(block, x)

    def check_full_network(self, device):
        torch.manual_seed(42)
        model = FIDNetMS(hid_S=8, hid_T=2).to(device)
        x = torch.randn(1, 5, 16, 20, device=device, requires_grad=True)
        self.assert_valid_backward(model, x)
        model.eval()
        with torch.no_grad():
            expected = model(x)
            restored = FIDNetMS(hid_S=8, hid_T=2).to(device)
            restored.load_state_dict(model.state_dict(), strict=True)
            restored.eval()
            torch.testing.assert_close(restored(x), expected)

    def test_full_network_cpu_backward_and_reload(self):
        self.check_full_network("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_full_network_cuda_backward_and_reload(self):
        self.check_full_network("cuda")


if __name__ == "__main__":
    unittest.main()
