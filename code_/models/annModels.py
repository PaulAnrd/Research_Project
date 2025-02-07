import numpy as np

##############################
# Activation Functions & Their Derivatives
##############################

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def identity(x):
    return x

def identity_deriv(x):
    return np.ones_like(x)

##############################
# Loss Functions and Metrics
##############################

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss"""
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    """Derivative of MSE loss w.r.t. predictions"""
    # Divide by number of examples in the batch to average the gradient
    return 2 * (y_pred - y_true) / y_true.shape[0]

def mae_metric(y_true, y_pred):
    """Mean Absolute Error metric"""
    return np.mean(np.abs(y_true - y_pred))

##############################
# Dense Layer Implementation
##############################

class Dense:
    def __init__(self, units, activation='relu', input_shape=None):
        """
        Parameters:
            units (int): Number of neurons.
            activation (str): Activation function ('relu' or 'linear').
            input_shape (tuple): Only needed for the first layer.
        """
        self.units = units
        self.activation_name = activation
        self.input_shape = input_shape  # if provided, used to initialize weights immediately
        self.weights = None
        self.biases = None
        self.cache = None   # to store inputs and linear outputs for backpropagation
        self.dW = None      # gradient w.r.t. weights
        self.db = None      # gradient w.r.t. biases
        
        # For Adam optimizer: initialize moment estimates later (once weights are set)
        self.m_W = None
        self.v_W = None
        self.m_b = None
        self.v_b = None
        
        # Set activation and its derivative
        if activation == 'relu':
            self.activation = relu
            self.activation_deriv = relu_deriv
        elif activation == 'linear' or activation is None:
            self.activation = identity
            self.activation_deriv = identity_deriv
        else:
            raise ValueError("Unsupported activation function: choose 'relu' or 'linear'")
    
    def initialize_weights(self, input_dim):
        """Initialize weights and biases given the input dimension."""
        # Use He initialization for relu; otherwise use a basic scaled initialization
        if self.activation_name == 'relu':
            self.weights = np.random.randn(input_dim, self.units) * np.sqrt(2. / input_dim)
        else:
            self.weights = np.random.randn(input_dim, self.units) * np.sqrt(1. / input_dim)
        self.biases = np.zeros((1, self.units))
        
        # Initialize Adam moment estimates (same shape as weights and biases)
        self.m_W = np.zeros_like(self.weights)
        self.v_W = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
    
    def forward(self, X):
        """
        Forward propagation for this layer.
        
        Parameters:
            X (np.ndarray): Input data of shape (batch_size, input_dim)
        
        Returns:
            A (np.ndarray): Output after applying the linear transformation and activation.
        """
        # Initialize weights if this is the first call (or if not pre-initialized)
        if self.weights is None:
            input_dim = X.shape[1]
            self.initialize_weights(input_dim)
        # Compute the linear part: Z = X.W + b
        Z = np.dot(X, self.weights) + self.biases
        # Apply activation: A = activation(Z)
        A = self.activation(Z)
        # Cache X and Z for use in backpropagation
        self.cache = (X, Z)
        return A
    
    def backward(self, dA):
        """
        Backward propagation for this layer.
        
        Parameters:
            dA (np.ndarray): Gradient of loss with respect to the layer's output.
        
        Returns:
            dX (np.ndarray): Gradient of loss with respect to the layer's input.
        """
        X, Z = self.cache
        # Compute the gradient after the activation function: dZ = dA * activation'(Z)
        dZ = dA * self.activation_deriv(Z)
        m = X.shape[0]  # batch size
        # Compute gradients of weights and biases
        self.dW = np.dot(X.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        # Compute gradient to pass to previous layer
        dX = np.dot(dZ, self.weights.T)
        return dX
    
    def update_params(self, t, lr, beta1, beta2, epsilon):
        """
        Update layer parameters using the Adam optimization algorithm.
        
        Parameters:
            t (int): Timestep (global batch counter).
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant to avoid division by zero.
        """
        # Update weights with Adam
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW ** 2)
        m_hat_W = self.m_W / (1 - beta1 ** t)
        v_hat_W = self.v_W / (1 - beta2 ** t)
        self.weights -= lr * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
        
        # Update biases with Adam
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.db
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db ** 2)
        m_hat_b = self.m_b / (1 - beta1 ** t)
        v_hat_b = self.v_b / (1 - beta2 ** t)
        self.biases -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

##############################
# Sequential Model Implementation
##############################

class Sequential:
    def __init__(self, layers):
        """
        Parameters:
            layers (list): A list of layer objects (e.g. Dense instances).
        """
        self.layers = layers
        self.compiled = False
    
    def compile(self, optimizer='adam', loss='mse', metrics=['mae'], lr=0.001):
        """
        Configures the model for training.
        
        Parameters:
            optimizer (str): Currently, only 'adam' is supported.
            loss (str): Loss function to use; only 'mse' is supported.
            metrics (list): List of metrics; supports ['mae'].
            lr (float): Learning rate.
        """
        if optimizer == 'adam':
            self.optimizer = 'adam'
            self.lr = lr
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
        else:
            raise ValueError("Only 'adam' optimizer is supported in this implementation.")
        
        if loss == 'mse':
            self.loss_fn = mse_loss
            self.loss_derivative = mse_loss_derivative
        else:
            raise ValueError("Only 'mse' loss is supported in this implementation.")
        
        if 'mae' in metrics:
            self.metric_fn = mae_metric
        else:
            raise ValueError("Only 'mae' metric is supported in this implementation.")
        
        self.t = 0  # Global timestep for Adam updates
        self.compiled = True
    
    def forward(self, X):
        """Run the forward pass through all layers."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.0, verbose=1):
        """
        Trains the model.
        
        Parameters:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
            epochs (int): Number of epochs to train.
            batch_size (int): Size of each mini-batch.
            validation_split (float): Fraction of data to use for validation.
            verbose (int): Verbosity mode.
            
        Returns:
            history (dict): Dictionary containing training (and optionally validation) loss and metric values per epoch.
        """
        if not self.compiled:
            raise Exception("Model must be compiled before training.")
        
        # Optionally split data into training and validation sets
        if validation_split > 0:
            permutation = np.random.permutation(X.shape[0])
            X = X[permutation]
            y = y[permutation]
            split_idx = int(X.shape[0] * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        history = {'loss': [], 'mae': []}
        if X_val is not None:
            history['val_loss'] = []
            history['val_mae'] = []
        
        for epoch in range(1, epochs + 1):
            # Shuffle training data at the start of each epoch
            permutation = np.random.permutation(num_samples)
            X_train = X_train[permutation]
            y_train = y_train[permutation]
            
            epoch_loss = 0
            epoch_mae = 0
            
            for batch in range(num_batches):
                start = batch * batch_size
                end = min(start + batch_size, num_samples)
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss and metric on the batch
                loss = self.loss_fn(y_batch, y_pred)
                mae = self.metric_fn(y_batch, y_pred)
                epoch_loss += loss * (end - start)
                epoch_mae += mae * (end - start)
                
                # Backward pass:
                # Compute gradient of loss w.r.t. predictions
                dA = self.loss_derivative(y_batch, y_pred)
                # Propagate gradients through layers (in reverse order)
                for layer in reversed(self.layers):
                    dA = layer.backward(dA)
                
                # Update parameters for each layer using Adam
                self.t += 1
                for layer in self.layers:
                    if hasattr(layer, 'update_params'):
                        layer.update_params(self.t, self.lr, self.beta1, self.beta2, self.epsilon)
            
            # Average loss and mae over the epoch
            epoch_loss /= num_samples
            epoch_mae /= num_samples
            
            msg = f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - mae: {epoch_mae:.4f}"
            
            # Run validation if requested
            if X_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_fn(y_val, y_val_pred)
                val_mae = self.metric_fn(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
                msg += f" - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}"
            
            history['loss'].append(epoch_loss)
            history['mae'].append(epoch_mae)
            if verbose:
                print(msg)
        
        return history
    
    def evaluate(self, X, y, verbose=1):
        """Evaluate the model on the given test data."""
        y_pred = self.forward(X)
        loss = self.loss_fn(y, y_pred)
        mae = self.metric_fn(y, y_pred)
        if verbose:
            print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
        return loss, mae
    
    def predict(self, X):
        """Generate output predictions for the input samples."""
        return self.forward(X)

##############################
# Example Usage
##############################

# Suppose you have your training and test data as NumPy arrays:
# X_train, y_train, X_test, y_test = ... (load or generate your data)

# You can create your model by replacing your tf.keras.layers with our Dense layers:
# For example, to replicate your original TensorFlow model:

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# You would write instead:

#model = Sequential([
#    Dense(64, activation='relu', input_shape=(None,)),  # input_shape is not used directly here except for initialization
#    Dense(32, activation='relu'),
#    Dense(16, activation='relu'),
#    Dense(1, activation='linear')  # linear activation for the final output
#])

# Compile the model (here, only 'adam' optimizer, 'mse' loss, and 'mae' metric are supported)
#model.compile(optimizer='adam', loss='mse', metrics=['mae'], lr=0.001)

# Train the model
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on test data
# loss, mae = model.evaluate(X_test, y_test, verbose=1)
# print(f"Test Loss: {loss}, Test MAE: {mae}")

# Make predictions
# predictions = model.predict(X_test)




class BatchNormalization:
    def __init__(self, momentum=0.9, epsilon=1e-5):
        """
        A minimal implementation of BatchNormalization in NumPy.
        
        Parameters:
            momentum (float): Momentum for running mean and variance.
            epsilon (float): Small constant to avoid division by zero.
        """
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = None  # scale parameter
        self.beta = None   # shift parameter
        self.running_mean = None
        self.running_var = None
        self.cache = None  # cache for backward pass
        
        # For Adam optimizer updates:
        self.m_gamma = None
        self.v_gamma = None
        self.m_beta = None
        self.v_beta = None

    def initialize_params(self, input_dim):
        """Initialize gamma, beta, and running averages for a given input dimension."""
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))

    def forward(self, X, training=True):
        """
        Forward pass for Batch Normalization.
        
        Parameters:
            X (np.ndarray): Input data of shape (batch_size, features).
            training (bool): If True, use batch statistics and update running averages.
            
        Returns:
            np.ndarray: Normalized and scaled output.
        """
        if self.gamma is None:
            self.initialize_params(X.shape[1])
            
        if training:
            # Compute batch statistics
            mu = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            X_norm = (X - mu) / np.sqrt(var + self.epsilon)
            out = self.gamma * X_norm + self.beta
            
            # Cache values for backpropagation
            self.cache = (X, X_norm, mu, var)
            
            # Update running averages
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var
        else:
            # Use running averages during inference
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * X_norm + self.beta
            
        return out

    def backward(self, d_out):
        """
        Backward pass for Batch Normalization.
        
        Parameters:
            d_out (np.ndarray): Gradient of the loss with respect to the output.
            
        Returns:
            np.ndarray: Gradient with respect to the input X.
        """
        X, X_norm, mu, var = self.cache
        N = X.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.epsilon)

        # Gradients of gamma and beta
        dgamma = np.sum(d_out * X_norm, axis=0, keepdims=True)
        dbeta  = np.sum(d_out, axis=0, keepdims=True)
        
        # Gradient w.r.t. normalized input
        dX_norm = d_out * self.gamma

        # Gradient w.r.t. variance
        dvar = np.sum(dX_norm * (X - mu) * (-0.5) * std_inv**3, axis=0, keepdims=True)
        # Gradient w.r.t. mean
        dmu = np.sum(dX_norm * (-std_inv), axis=0, keepdims=True) + dvar * np.mean(-2 * (X - mu), axis=0, keepdims=True)
        # Gradient w.r.t. X
        dX = dX_norm * std_inv + dvar * 2 * (X - mu) / N + dmu / N

        # Store gradients for parameter updates
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dX

    def update_params(self, t, lr, beta1, beta2, epsilon):
        """
        Update parameters using the Adam optimization algorithm.
        
        Parameters:
            t (int): Timestep.
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant to avoid division by zero.
        """
        # Initialize moment variables if not already done
        if self.m_gamma is None:
            self.m_gamma = np.zeros_like(self.gamma)
            self.v_gamma = np.zeros_like(self.gamma)
            self.m_beta = np.zeros_like(self.beta)
            self.v_beta = np.zeros_like(self.beta)

        # Update gamma parameters
        self.m_gamma = beta1 * self.m_gamma + (1 - beta1) * self.dgamma
        self.v_gamma = beta2 * self.v_gamma + (1 - beta2) * (self.dgamma ** 2)
        m_hat_gamma = self.m_gamma / (1 - beta1 ** t)
        v_hat_gamma = self.v_gamma / (1 - beta2 ** t)
        self.gamma -= lr * m_hat_gamma / (np.sqrt(v_hat_gamma) + epsilon)

        # Update beta parameters
        self.m_beta = beta1 * self.m_beta + (1 - beta1) * self.dbeta
        self.v_beta = beta2 * self.v_beta + (1 - beta2) * (self.dbeta ** 2)
        m_hat_beta = self.m_beta / (1 - beta1 ** t)
        v_hat_beta = self.v_beta / (1 - beta2 ** t)
        self.beta -= lr * m_hat_beta / (np.sqrt(v_hat_beta) + epsilon)
