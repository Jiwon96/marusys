import torch

class Padding:
    """zero pad to have 1 sec len"""

    def __init__(self):
        self.SR = 16000
        self.output_len = self.SR * 2

    def __call__(self, x):
        pad_len = self.output_len - x.shape[-1]
        if pad_len > 0:
            x = torch.cat([x, torch.zeros([x.shape[0], pad_len])], dim=-1)
        elif pad_len < 0:
            raise ValueError("no sample exceed 1sec in GSC.")
        return x