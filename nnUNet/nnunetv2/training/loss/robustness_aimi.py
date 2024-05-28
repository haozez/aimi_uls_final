import torch
from torch import nn

class Robustness_AIMI(nn.Module):
    ''' Loss term penalizing model predictions differing when the same
    input is passed through the model'''
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # NB that x contains the results of forward passing the rotated input
        # through the model already; it's got a shape that is two times 
        # the batch size in dimension 0.
        batch_size = x.shape[0] // 2
        unrotated = x[:batch_size,...] # First half is the original input
        # Unrotate the rotated output to match the original
        rotated = torch.rot90(x[batch_size:, ...], k=2, dims=[3,4])
        # Find the absolute value of the difference between the outputs.
        # Should work regardless of what the model outputs (logits or not).
        # Finds the differences individually for each example within the batch. 
        total_difference = torch.sum(torch.abs(unrotated-rotated), axis=(1,2,3,4))
        # Divide by the number of pixels in each VOI, normalizing the terms
        num_pixels = torch.prod(torch.tensor(unrotated[0].shape)).item()
        total_difference *= 1/num_pixels

        # Take mean across batch
        total_difference = torch.mean(total_difference, axis=0)

        return total_difference


    