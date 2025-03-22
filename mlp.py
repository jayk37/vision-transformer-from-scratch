import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        
        # Compute Z1 As Input To First Layer
        z1 = torch.matmul(x, self.parameters['W1'].t()) + self.parameters['b1']
        
        # Apply Activation Function (f_function) To First Layer Output
        if self.f_function == "relu":
            a1 = torch.relu(z1)
        elif self.f_function == "sigmoid":
            a1 = 1.0 / (1 + torch.exp(-z1))
        else:
            a1 = z1
        
        # Cache Intermediate Values For Backpropagation
        self.cache['x'] = x
        self.cache['z1'] = z1
        self.cache['a1'] = a1

        # Compute Z2 As Input To Second Layer
        z2 = torch.matmul(a1, self.parameters['W2'].t()) + self.parameters['b2']

        # Apply Activation Function (g_function) To Second Layer Output
        if self.g_function == "relu":
            y_hat = torch.relu(z2)
        elif self.g_function == "sigmoid":
            y_hat = 1.0 / (1 + torch.exp(-z2))
        else:
            y_hat = z2
        
        # Cache The Final Output (y_hat) And Intermediate z2 For Backpropagation
        self.cache['z2'] = z2
        self.cache['y_hat'] = y_hat

        return y_hat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        
        # Retrieve Cached Values From Forward Pass
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        y_hat = self.cache['y_hat']

        # Compute Gradient Of z2 Based On g_function (Second Layer Activation)
        if self.g_function == "relu":
            dz2 = dJdy_hat.clone()
            dz2 [ z2 <= 0 ] = 0
        elif self.g_function == "sigmoid":
            dz2 = dJdy_hat * y_hat * (1 - y_hat)
        else:
            dz2 = dJdy_hat

        # Compute Gradients For W2 And b2 (Second Layer)
        self.grads['dJdW2'] += torch.matmul(dz2.t(), a1)
        self.grads['dJdb2'] += dz2.sum(0)

        # Compute Gradient Of a1 (Output Of First Layer) With Respect To Loss
        da1 = torch.matmul(dz2, self.parameters['W2'])

        # Compute Gradient Of z1 Based On f_function (First Layer Activation)
        if self.f_function == "relu":
            dz1 = da1.clone()
            dz1 [ z1 <= 0 ] = 0
        elif self.f_function == "sigmoid":
            dz1 = da1 * (a1) * (1-a1)
        else:
            dz1 = da1
        
        # Compute Gradients For W1 And b1 (First Layer)
        self.grads['dJdW1'] += torch.matmul(dz1.t(), x)
        self.grads['dJdb1'] += dz1.sum(0)

    def clear_grad_and_cache(self):
        # Clear Gradients After Each Backpropagation Step
        for grad in self.grads:
            self.grads[grad].zero_()

        # Clear Cached Values
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """

    # Compute Mean Squared Error Loss
    m = y.size(1) * y.size(0)
    J = ((y_hat - y) ** 2).mean()

    # Compute Gradient Of Loss With Respect To Predictions
    dJdy_hat = 2 * (y_hat - y) / m
    return J, dJdy_hat


def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    
    # Compute Binary Cross-Entropy Loss With Clamping To Prevent Log(0)
    eps = 1e-8
    m = (y.size(0) * y.size(1))
    y_hat_clamped = torch.clamp(y_hat, eps, 1 - eps)
    loss = - (y * torch.log(y_hat_clamped) + (1 - y) * torch.log(1 - y_hat_clamped))
    loss = loss.mean()

    # Compute Gradient Of BCE Loss With Respect To Predictions
    dJdy_hat = (y_hat - y) / (y_hat_clamped * (1 - y_hat_clamped)) / m
    return loss, dJdy_hat

