import numpy as np

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons, wlambal1=0, wlambdal2=0, blambdal2=0, blambdal1=0):
        #initializing weights
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.lambdaw1 = wlambal1
        self.lambdaw2 = wlambdal2
        self.lambdab1 = blambdal1
        self.lambdab2 = blambdal2

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        self.dweights = np.dot(np.transpose(self.inputs), dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.lambdaw1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights<0] = -1
            self.dweights += self.lambdaw1*dL1 #conversion to absolute
        if self.lambdaw2 > 0:
            self.dweights += 2*self.lambdaw2*self.weights
        if self.lambdab1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases<0] = -1
            self.dbiases += self.lambdab1*dL1
        if self.lambdab2 > 0:
            self.dbiases += 2*self.lambdab2*self.biases

        self.dinputs = np.dot(dvalues, np.transpose(self.weights))

class Dropout_Layer:
    def __init__(self, rate):
        self.rate = 1 - rate
        self.binary_mask = None
    
    def forward(self, inputs, training=True):
        self.inputs = inputs
        if training:
            # Generate mask only during training
            self.binary_mask = (np.random.rand(*inputs.shape) < self.rate) / self.rate
            self.output = inputs * self.binary_mask
        else:
            # No dropout during testing, just scale the output
            self.output = inputs * self.rate
    
    def backward(self, dvalues):
        if self.binary_mask is not None:
            self.dinputs = dvalues * self.binary_mask
        else:
            self.dinputs = dvalues * self.rate

class ReLU_Activation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Softmax_Activation:
    def forward(self, inputs):
        values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = values/np.sum(values, axis=1, keepdims=True)
        self.output = probabilities

class Sigmoid_Activation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

class Loss:
    def calculate(self, input, target):
        sample_losses = self.forward(input, target)
        loss = np.mean(sample_losses)
        
        return loss
    
    def regularization(self, layer):
        reg_loss = 0

        if layer.lambdaw1 > 0:
            reg_loss += layer.lambdaw1 * np.sum(np.abs(layer.weights))
        if layer.lambdaw2 > 0:
            reg_loss += layer.lambdaw2 * np.sum(layer.weights*layer.weights)
        if layer.lambdab1 > 0:
            reg_loss += layer.lambdab1 * np.sum(np.abs(layer.biases))
        if layer.lambdab2 > 0:
            reg_loss += layer.lambdab2 * np.sum(layer.biases*layer.biases)
        
        return reg_loss

class Entropy_Loss(Loss):
    def forward(self, predicted, target):
        n_samples = len(predicted)
        clipped_pred = np.clip(predicted, 1e-7, 1-1e-7)

        if len(target.shape) == 1:
            predictives = clipped_pred[range(n_samples), target]
        elif len(target.shape) == 2:
            predictives = np.sum(clipped_pred*target, axis=1)
        
        probabilities = -np.log(predictives)

        return np.mean(probabilities)
    
    def backward(self, dvalues, target):
        n_sample = len(dvalues)

        labels = len(dvalues[0])

        if len(target.shape) == 1:
            target = np.eye(labels)[target]
        
        self.dinputs = -target/dvalues
        self.dinputs /= n_sample
    
class Combo_Loss_Softmax():
    def __init__(self):
        self.activation = Softmax_Activation()
        self.loss = Entropy_Loss()

    def forward(self, inputs, target):
        self.activation.forward(inputs)
        self.output = self.activation.output

        self.loss.forward(self.output, target)

        return self.loss.calculate(self.output, target)
    
    def backward(self, dvalues, target):
        n_sample = len(dvalues)
        if len(target.shape) == 2:
            target = np.argmax(target, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(n_sample), target] -= 1

        self.dinputs /= n_sample

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay = 0.1, momentum=0.0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.momentum = momentum

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay*self.iterations))
    
    def update(self, layer):
        if self.momentum:
            if not hasattr(layer, 'momentum_w'):
                layer.momentum_w = np.zeros_like(layer.weights)
                layer.momentum_b = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.momentum_w - self.current_learning_rate * layer.dweights
            layer.momentum_w = weight_updates
            bias_updates = self.momentum * layer.momentum_b - self.current_learning_rate * layer.dbiases
            layer.momentum_b = bias_updates
        else:    
            weight_updates = self.learning_rate*layer.dweights
            bias_updates = self.learning_rate*layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update(self):
        self.iterations += 1

class RMS_Optimizer:
    def __init__(self, learning_rate=0.001, decay = 0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay*self.iterations))
    
    def update(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho)*layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho)*layer.dbiases**2

        layer.weights -= self.current_learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases -= self.current_learning_rate*layer.dbiases/(np.sqrt(layer.bias_cache)+self.epsilon)

    def post_update(self):
        self.iterations += 1

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, learning_rate=0.001, decay=0.0, epsilon=1e-7):
        self.decay = decay
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay*self.iterations))

    def update(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.momentum_w = np.zeros_like(layer.weights)

            layer.bias_cache = np.zeros_like(layer.biases)
            layer.momentum_b = np.zeros_like(layer.biases)

        layer.momentum_w = self.beta1*layer.momentum_w + (1-self.beta1)*layer.dweights
        layer.momentum_b = self.beta1*layer.momentum_b + (1-self.beta1)*layer.dbiases

        corr_momentum_w = layer.momentum_w / (1 - self.beta1**(self.iterations+1))
        corr_momentum_b = layer.momentum_b / (1 - self.beta1**(self.iterations+1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (1-self.beta2)*layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2)*layer.dbiases**2

        corr_cache_w = layer.weight_cache / (1 - self.beta2 ** (self.iterations+1))
        corr_cache_b = layer.bias_cache / (1 - self.beta2 ** (self.iterations+1))

        layer.weights -= self.current_learning_rate*corr_momentum_w/(np.sqrt(corr_cache_w)+self.epsilon)
        layer.biases -= self.current_learning_rate*corr_momentum_b/(np.sqrt(corr_cache_b)+self.epsilon)

    def post_update(self):
        self.iterations += 1

def build_network():
    mapping = {
        "D": Dense_Layer,
        "R": ReLU_Activation,
        "S": Combo_Loss_Softmax,
        "d": Dropout_Layer,
        "OS": Optimizer_SGD,
        "OR": RMS_Optimizer,
        "OA": Adam
    }

    layers = []

    print("""Define structure of the network:
          D => Dense Layer,
          R => ReLU Activation Layer,
          S => Softmax Activation Layer,
          d => Dropout Layer,
          OS => SGD Optimizer,
          OR => RMS Optimizer,
          OA => Adam Optimizer
          {Optimizers to be mentioned at the end}
          Terminate network with TERM
          
          Example: D_R_d_D_S_TERM_OS""")
    order = input()

    for ch in order.split("_"):
        layers.append(mapping[ch])
    
    return layers

def pass_forward(layers, inputs, target):
    prev_layer = layers[0]
    prev_layer.forward(inputs)
    for curr_layer in layers[1:]:
        if isinstance(curr_layer) is Combo_Loss_Softmax:
            curr_layer.forward(prev_layer.output, target)
        else:
            curr_layer.forward(prev_layer.output)
        prev_layer = curr_layer

    return None

def compute_loss(layers, target):
    dense = []
    optimizer = None
    for layer in layers:
        if isinstance(layer) is Combo_Loss_Softmax:
            softmax = layer
        elif isinstance(layer) is Dense_Layer:
            dense.append(layer)
        elif isinstance(layer) is Adam or isinstance(layer) is RMS_Optimizer or isinstance(layer) is Optimizer_SGD:
            optimizer = layer
    
    base_loss = softmax.loss.calculate(softmax.output, target)
    reg_loss = 0
    for layer in dense:
        reg_loss += softmax.loss.regularization(layer)
    
    epoch_loss = base_loss + reg_loss

    return epoch_loss, dense

def pass_backward(layers, target):
    l = len(layers)
    i = l-2

    next_layer = layers[i+1]
    if isinstance(next_layer) is Combo_Loss_Softmax:
        next_layer.backward(next_layer.output, target)
    elif isinstance(next_layer) is Dense_Layer:
        next_layer.backward(next_layer.output)

    while i >= 0:
        curr_layer = layers[i]
        curr_layer.backward(next_layer.dinputs)
        next_layer = curr_layer

        i -= 1
    
    return None

def optimize(dense, optimizer):
    optimizer.pre_update()
    for layer in dense[::-1]:
        optimizer.update(layer)
    optimize.post_update()

