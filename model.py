import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        
        # Layer dimensions
        self.input_dim = 384
        self.hidden_dim = 128
        self.output_dim = 2
        
        # Define the network layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
    def initialize_weights(self):
        # Initialize weights using Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Forward pass through the network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, hidden_dims=[256, 512, 256]):
        super(FullyConnectedNetwork, self).__init__()
        
        # Create list to hold all layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # Final hidden layer to output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolutional layer
        # in_channels=1 (assuming grayscale input)
        # out_channels=64 (number of kernels)
        # kernel_size=3 (3x3 kernel)
        # stride=1
        self.conv1 = nn.Conv2d(in_channels=1, 
                              out_channels=64, 
                              kernel_size=3, 
                              stride=1,
                              padding=0)  # padding to maintain spatial dimensions
        
        # Calculate the flattened size for the fully connected layer
        # This will depend on your input image size
        # For example, if input is 26x26:
        # After conv1: 26x26x64
        self.flatten_size = 26 * 26 * 64
        
        # Fully connected layer with 128 output features
        self.fc1 = nn.Linear(self.flatten_size, 128)
        
    def forward(self, x):
        # Apply convolution and ReLU activation
        x = F.relu(self.conv1(x))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, self.flatten_size)
        
        # Apply fully connected layer
        x = self.fc1(x)
        
        return x

# Example usage:
def main():
    # Create model instance
    conv_model = ConvNet()
    fc_model = FullyConnectedNetwork()
    q_net = Qnet()
    # Example input (batch_size=1, channels=1, height=28, width=28)
    example_input_left = torch.randn(1, 28, 28)
    example_input_right = torch.randn(1, 28, 28)
    fully_connected_input = torch.randn(1, 128)
    
    # Forward pass

    
    # Forward pass
    output_left = conv_model(example_input_left) # Left conv layer
    output_right = conv_model(example_input_right)
    output_fc = fc_model(fully_connected_input)

     
    # Cascade the inputs
    input_q_net = torch.cat((output_left,output_right,output_fc),dim=1)
    q_net_output = q_net(input_q_net)
    print(input_q_net.shape)
    print(q_net_output.shape)
    print(f"Output shape: {output_left.shape}")  # Should be torch.Size([1, 128])

if __name__ == "__main__":
    main()
