from torch import nn
from GELU import GELU
import torch


class ExampleDeepNeuralNetwork(nn.Module):
    """
    5-layer network demonstrating how shortcut (residual) connections prevent vanishing gradients.
    """
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = (nn.ModuleList(
            [nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
             nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
             nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
             nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
             nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
             ]))

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x) # Compute the output of the current layer
            # Check if shortcut can be applied
            if self.use_shortcut and layer_output.shape == x.shape:
                x = layer_output + x
            else:
                x = layer_output
        return x

def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

layer_sizes = [3, 3, 3, 3, 3, 1]

sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

print_gradients(model_without_shortcut, sample_input)
print_gradients(model_with_shortcut, sample_input)

# layers.0.0.weight has gradient mean of 0.00020173587836325169
# layers.1.0.weight has gradient mean of 0.0001201116101583466
# layers.2.0.weight has gradient mean of 0.0007152041653171182
# layers.3.0.weight has gradient mean of 0.001398873864673078
# layers.4.0.weight has gradient mean of 0.005049646366387606
# layers.0.0.weight has gradient mean of 0.0014432319439947605
# layers.1.0.weight has gradient mean of 0.004846962168812752
# layers.2.0.weight has gradient mean of 0.0041389018297195435
# layers.3.0.weight has gradient mean of 0.00591512955725193
# layers.4.0.weight has gradient mean of 0.03265950828790665



