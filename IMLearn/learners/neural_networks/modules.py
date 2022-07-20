import numpy as np
from IMLearn.base.base_module import BaseModule
from IMLearn.metrics.loss_functions import cross_entropy, softmax


class FullyConnectedLayer(BaseModule):
    """
    Module of a fully connected layer in a neural network

    Attributes:
    -----------
    input_dim_: int
        Size of input to layer (number of neurons in preceding layer

    output_dim_: int
        Size of layer output (number of neurons in layer_)

    activation_: BaseModule
        Activation function to be performed after integration of inputs and weights

    weights: ndarray of shape (input_dim_, outout_din_)
        Parameters of function with respect to which the function is optimized.

    include_intercept: bool
        Should layer include an intercept or not
    """

    def __init__(self, input_dim: int, output_dim: int,
                 activation: BaseModule = None,
                 include_intercept: bool = True):
        """
        Initialize a module of a fully connected layer

        Parameters:
        -----------
        input_dim: int
            Size of input to layer (number of neurons in preceding layer

        output_dim: int
            Size of layer output (number of neurons in layer_)

        activation_: BaseModule, default=None
            Activation function to be performed after integration of inputs and weights. If
            none is specified functions as a linear layer

        include_intercept: bool, default=True
            Should layer include an intercept or not

        Notes:
        ------
        Weights are randomly initialized following N(0, 1/input_dim)
        """

        super().__init__()
        self.weights = np.random.normal(0, 1 / input_dim,
                                        size=(input_dim + 1, output_dim)) \
            if include_intercept else np.random.normal(0, 1 / input_dim, size=(
            input_dim, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.include_intercept = include_intercept

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute activation(weights @ x) for every sample x: output value of layer at point
        self.weights and given input

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights

        Returns:
        --------
        output: ndarray of shape (n_samples, output_dim)
            Value of function at point self.weights
        """
        if self.include_intercept:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        # if not self.activation:
        #     return X @ self.weights
        # return self.activation.compute_output(X=X @ self.weights)

        output = X @ self.weights

        # if self.activation is not None:
        output = self.activation.compute_output(X=output, **kwargs)
        return output

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point self.weights

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights

        Returns:
        -------
        output: ndarray of shape (input_dim, n_samples)
            Derivative with respect to self.weights at point self.weights
        """
        if self.include_intercept:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        if not self.activation:
            # return np.identity(X.shape[1])
            return self.weights

        return self.activation.compute_jacobian(X=X @ self.weights) @ X.T


class ReLU(BaseModule):
    """
    Module of a ReLU activation function computing the element-wise function ReLU(x)=max(x,0)
    """

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute element-wise value of activation

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be passed through activation

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            Data after performing the ReLU activation function
        """
        X[X < 0] = 0
        return X

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to given data

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to compute derivative with respect to

        Returns:
        -------
        output: ndarray of shape (n_samples,)
            Element-wise derivative of ReLU with respect to given data
        """
        X[X < 0] = 0
        X[X > 0] = 1
        return X


class CrossEntropyLoss(BaseModule):
    """
    Module of Cross-Entropy Loss: The Cross-Entropy between the Softmax of a sample x and e_k for a true class k
    """

    def compute_output(self, X: np.ndarray, y: np.ndarray,
                       **kwargs) -> np.ndarray:
        """
        Computes the Cross-Entropy over the Softmax of given data, with respect to every

        CrossEntropy(Softmax(x),e_k) for every sample x

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data for which to compute the cross entropy loss

        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed

        Returns:
        --------
        output: ndarray of shape (n_samples,)
            cross-entropy loss value of given X and y
        """
        sm = softmax(X)
        loss = []
        for i in range(X.shape[0]):
            loss.append(cross_entropy(y[i], sm[i]))
        return np.array(loss)
        # e_k = np.zeros_like(X)
        # for i in range(len(y)):
        #     e_k[i][y[i]] = 1
        # for i in range(X.shape[0]):
        #     loss.append(cross_entropy(sm[i], e_k))
        # return np.array(loss)

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray,
                         **kwargs) -> np.ndarray:
        """
        Computes the derivative of the cross-entropy loss function with respect to every given sample

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data with respect to which to compute derivative of the cross entropy loss

        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            derivative of cross-entropy loss with respect to given input
        """
        sm = softmax(X)
        e_k = np.zeros_like(X)
        for i in range(len(y)):
            e_k[i][y[i]] = 1
        return sm - e_k
