import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet_Base(nn.Module):
    def __init__(
        self,
        # --- Convolutional branches ---
        in_channels_left: int = 1,
        in_channels_right: int = 1,
        conv_out_channels: int = 64,
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        # If you know your input height/width, you can set them here 
        # to automate computing the Flatten dimension:
        left_input_shape=(1, 84, 84),   # (channels, height, width)
        right_input_shape=(1, 84, 84),
        conv_activation=nn.ReLU,
        conv_output_dim: int = 128,  # final FC output size for each conv branch

        # --- Fully-connected input branch ---
        fc_input_dim: int = 32,      # dimension of train-properties vector 'i'
        fc_hidden_dim: int = 128,    # hidden dim for the train-properties branch
        fc_activation=nn.ReLU,
        fc_output_dim: int = 128,    # final output size of the FC branch

        # --- Q-value network ---
        q_hidden_dim: int = 128,     # hidden layer size in Q-value network
        q_output_dim: int = 2,       # number of actions (e.g., Move/Halt)
        q_activation=nn.ReLU
    ):
        super(QNet_Base, self).__init__()

        # ----------------------------------------------------------------------
        # 1) Build the Left Convolutional Branch
        # ----------------------------------------------------------------------
        self.left_conv = nn.Conv2d(
            in_channels_left,
            conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride
        )
        self.left_conv_activation = conv_activation()

        # We'll determine the size after the single conv layer so we can
        # build the linear layer to 128 out. We do this by passing a dummy
        # input through the conv and flattening.
        dummy_left = torch.zeros(1, *left_input_shape)
        left_conv_out = self.left_conv_activation(
            self.left_conv(dummy_left)
        )
        left_conv_out_size = left_conv_out.view(1, -1).size(1)

        self.left_linear = nn.Linear(left_conv_out_size, conv_output_dim)

        # ----------------------------------------------------------------------
        # 2) Build the Right Convolutional Branch
        # ----------------------------------------------------------------------
        self.right_conv = nn.Conv2d(
            in_channels_right,
            conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride
        )
        self.right_conv_activation = conv_activation()

        dummy_right = torch.zeros(1, *right_input_shape)
        right_conv_out = self.right_conv_activation(
            self.right_conv(dummy_right)
        )
        right_conv_out_size = right_conv_out.view(1, -1).size(1)

        self.right_linear = nn.Linear(right_conv_out_size, conv_output_dim)

        # ----------------------------------------------------------------------
        # 3) Build the FC input branch for train properties (vector i)
        # ----------------------------------------------------------------------
        self.fc_input = nn.Linear(fc_input_dim, fc_hidden_dim)
        self.fc_input_activation = fc_activation()
        self.fc_output = nn.Linear(fc_hidden_dim, fc_output_dim)

        # ----------------------------------------------------------------------
        # 4) Build the Q-value network
        #    The combined input is 3 * 128 = 384, by default
        # ----------------------------------------------------------------------
        q_in_dim = conv_output_dim + conv_output_dim + fc_output_dim
        self.q_hidden = nn.Linear(q_in_dim, q_hidden_dim)
        self.q_activation = q_activation()
        self.q_out = nn.Linear(q_hidden_dim, q_output_dim)

    def forward(self, left_img, right_img, train_props):
        """
        Args:
            left_img:   Tensor of shape (B, in_channels_left, H, W)
            right_img:  Tensor of shape (B, in_channels_right, H, W)
            train_props: Tensor of shape (B, fc_input_dim)
        Returns:
            Q-values of shape (B, q_output_dim)
        """

        # Left branch
        x_left = self.left_conv_activation(self.left_conv(left_img))
        x_left = x_left.view(x_left.size(0), -1)  # Flatten
        x_left = self.left_linear(x_left)

        # Right branch
        x_right = self.right_conv_activation(self.right_conv(right_img))
        x_right = x_right.view(x_right.size(0), -1)  # Flatten
        x_right = self.right_linear(x_right)

        # FC branch for train properties
        x_fc = self.fc_input_activation(self.fc_input(train_props))
        x_fc = self.fc_output(x_fc)

        # Concatenate
        x_combined = torch.cat([x_left, x_right, x_fc], dim=1)

        # Q-value network
        x_q = self.q_activation(self.q_hidden(x_combined))
        q_values = self.q_out(x_q)

        return q_values


if __name__ == "__main__":
    # Example usage:

    # Instantiate the network
    net = QNet_Base(
        in_channels_left=1,
        in_channels_right=1,
        left_input_shape=(1, 84, 84),
        right_input_shape=(1, 84, 84),
        fc_input_dim=10,  # example if train_props has length 10
    )

    # Dummy inputs
    batch_size = 4
    dummy_left_img = torch.randn(batch_size, 1, 84, 84)   # (B, C, H, W)
    dummy_right_img = torch.randn(batch_size, 1, 84, 84)  # (B, C, H, W)
    dummy_train_props = torch.randn(batch_size, 10)       # (B, fc_input_dim)

    # Forward pass
    q_vals = net(dummy_left_img, dummy_right_img, dummy_train_props)
    print("Output Q-values shape:", q_vals.shape)  # Should be [4, 2]
    print(q_vals)
