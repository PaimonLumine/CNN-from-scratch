import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import json

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test)  = mnist.load_data()
x = x_train.reshape(-1, 28 * 28) 
x = (x-x.mean(axis=1).reshape(-1, 1))/x.std(axis=1).reshape(-1, 1)
x = x.reshape(-1, 28, 28, 1) 
y = pd.get_dummies(y_train).to_numpy()
xt = x_test.reshape(-1, 28 * 28) 
xt = (xt-xt.mean(axis=1).reshape(-1, 1))/xt.std(axis=1).reshape(-1, 1)
xt = xt.reshape(-1, 28, 28, 1) 
yt = pd.get_dummies(y_test).to_numpy()


class FFL(object):
    def __init__(self, input_shape = None, neurons = 1, bias = None, weights=None, activation=None, is_bias = True):
        np.random.seed(100)
        self.input_shape = input_shape
        self.neurons = neurons
        self.isbias = is_bias
        self.name = ""
        self.w = weights
        self.b = bias
        if input_shape != None:
            self.output_shape = neurons
        if self.input_shape != None:
            self.weights = weights if weights != None else np.random.randn(self.input_shape, neurons)
            self.parameters = self.input_shape *  self.neurons + self.neurons if self.isbias else 0  
        if(is_bias):
            self.biases = bias if bias != None else np.random.randn(neurons)
        else:
            self.biases = 0  
        self.out = None
        self.input = None
        self.error = None
        self.delta = None
        activations = ["Linear", "relu", "sigmoid", "tanh", "softmax"]
        self.delta_weights = 0
        self.delta_biases = 0
        self.pdelta_weights = 0
        self.pdelta_biases = 0        
        if activation not in activations and activation != None:
             raise ValueError(f"Activation function not recognised. Use one of {activations} instead.")
        else:
            self.activation = activation
        if self.activation == None:
            self.activation = "linear"

    def activation_fn(self, r):
        """
        A method of FFL which contains the operation and defination of given activation function.
        """
        if self.activation == None or self.activation == "linear":
            return r
        if self.activation == "relu":
            r[r<0] = 0
            return r
        elif self.activation == 'tanh': #tanh
            return np.tanh(r)
        elif self.activation == 'sigmoid':  # sigmoid
            return 1 / (1 + np.exp(-r))
        elif self.activation == "softmax":# stable softmax   
            r = r - np.max(r)
            s = np.exp(r)
            return s / np.sum(s)

    def activation_dfn(self, r):
        """
        A method of FFL to find derivative of given activation function.
        """
        if self.activation == None or self.activation == "linear":
            return np.ones(r.shape)
        elif self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == "relu":
            r[r<0] = 0
            return r
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        elif self.activation == 'softmax':
            soft = self.activation_fn(r)
            return soft * (1 - soft)

    def apply_activation(self, x):
        soma = np.dot(x, self.weights) + self.biases
        self.out = self.activation_fn(soma)
        return self.out

    def set_n_input(self):
        self.weights = self.w if self.w != None else np.random.normal(size=(self.input_shape, self.neurons))

    def get_parameters(self):
        self.parameters = self.input_shape * self.neurons + self.neurons if self.isbias else 0
        return self.parameters

    def set_output_shape(self):
        self.set_n_input()
        self.output_shape = self.neurons
        self.get_parameters()

    def backpropagate(self, nx_layer):
        self.error = np.dot(nx_layer.weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfn(self.out)
        self.delta_weights += self.delta * np.atleast_2d(self.input).T
        self.delta_biases += self.delta

class Conv2d():
    def __init__(self, input_shape = None, filters = 1, kernel_size = (3,3), isbias = True, activation = None, stride = (1, 1), padding = "zero", kernel = None, bias = None):
        self.input_shape = input_shape
        self.filters = filters
        self.isbias = isbias
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.p = 1 if padding != None else 0
        self.bias = bias
        self.kernel = kernel
        if input_shape != None:
            self.kernel_size = (kernel_size[0], kernel_size[1], input_shape[2], filters)
            self.output_shape = (int((input_shape[0] - kernel_size[0] + 2 * self.p) / stride[0]) + 1,
                                 int((input_shape[1] - kernel_size[1] + 2 * self.p) / stride[0]) + 1, filters)
            self.set_variables()
            self.out = np.zeros(self.output_shape)
        else:
            self.kernel_size = (kernel_size[0], kernel_size[1])
            
    def set_variables(self):
        self.weights = self.init_param(self.kernel_size)
        self.biases = self.init_param((self.filters, 1))
        self.parameters = np.multiply.reduce(self.kernel_size) + self.filters if self.isbias else 1
        self.delta_weights = np.zeros(self.kernel_size)
        self.delta_biases = np.zeros(self.biases.shape)

    def init_param(self, size):
        stddev = 1/np.sqrt(np.prod(size))
        return np.random.normal(loc=0, scale=stddev, size=size)

    def activation_fn(self, r):
        if self.activation == None or self.activation == "linear":
            return r   
        if self.activation == "relu":
            r[r<0] = 0
            return r
        elif self.activation == 'tanh': #tanh
            return np.tanh(r)
        elif self.activation == 'sigmoid':  # sigmoid
            return 1 / (1 + np.exp(-r))
        elif self.activation == "softmax":# stable softmax   
            r = r - np.max(r)
            s = np.exp(r)
            return s / np.sum(s)

    def activation_dfn(self, r):
        if self.activation == None or self.activation == "linear":
            return np.ones(r.shape)
        elif self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == "relu":
            r[r<0] = 0
            return r
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        elif self.activtion == 'softmax':
            soft = self.activation_fn(r)
            return soft * (1 - soft)

    def apply_activation(self, image):
        for f in range(self.filters):
            image = self.input
            kshape = self.kernel_size
            if kshape[0] % 2 != 1 or kshape[1] % 2 != 1:
                raise ValueError("Please provide odd length of 2d kernel")
            if type(self.stride) == int:
                stride == (stride, stride)
            else:
                stride = self.stride
            shape = image.shape
            if self.padding == "zero":
                zeros_h = np.zeros((shape[1], shape[2])).reshape(-1, shape[1], shape[2])
                zeros_v = np.zeros((shape[0] + 2, shape[2])).reshape(shape[0] + 2, -1, shape[2])
                padded_img = np.vstack((zeros_h, image, zeros_h))
                padded_img = np.hstack((zeros_v, padded_img, zeros_v))
                image = padded_img
                shape = image.shape
            elif self.padding == "same":
                h1 = image[0].reshape(-1, shape[1], shape[2])
                h2 = image[-1].reshape(-1, shape[1], shape[2])
                padded_img = np.vstack((h1, image, h2))
                v1 = padded_img[:, 0].reshape(padded_img.shape[0], -1, shape[2])
                v2 = padded_img[:, -1].reshape(padded_img.shape[0], -1, shape[2])
                padded_img = np.hstack((v1, padded_img, v2))
                image = padded_img
                shape = image.shape
            elif self.padding == None:
                pass
            rv = 0
            cimg = []
            for r in range(kshape[0], shape[0] + 1, stride[0]):
                cv = 0
                for c in range(kshape[1], shape[1] + 1, stride[1]):
                    chunk = image[rv:r, cv:c]
                    soma = (np.multiply(chunk, self.weights[:,:,:,f]))
                    summa = soma.sum() + self.biases[f]
                    cimg.append(summa)
                    cv += stride[1]
                rv += stride[0]
            cimg = np.array(cimg).reshape(int(rv / stride[0]), int(cv / stride[1]))
            self.out[:, :, f] = cimg
        self.out = self.activation_fn(self.out)
        return self.out


    def set_output_shape(self):
        #print(self.input_shape, self.kernel_size, self.stride)
        self.kernel_size = (self.kernel_size[0], self.kernel_size[1], self.input_shape[2], self.filters)
        self.set_variables()
        self.output_shape = (int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0] + 1), 
                                int((self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1] + 1), self.filters)
        self.out = np.zeros(self.output_shape)
        self.dout = np.zeros((self.output_shape[0], self.output_shape[1], self.input_shape[2], self.output_shape[2]))


    def backpropagate(self, nx_layer):
        layer = self
        layer.delta = np.zeros((layer.input_shape[0], layer.input_shape[1], layer.input_shape[2]))
        image = layer.input
        for f in range(layer.filters):
            kshape = layer.kernel_size
            shape = layer.input_shape
            stride = layer.stride
            rv = 0
            i = 0
            for r in range(kshape[0], shape[0] + 1, stride[0]):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1] + 1, stride[1]):
                    chunk = image[rv:r, cv:c]
                    layer.delta_weights[:, :, :, f] += chunk * nx_layer.delta[i, j, f]
                    layer.delta[rv:r, cv:c, :] += nx_layer.delta[i, j, f] * layer.weights[:, :, :, f]
                    j += 1
                    cv += stride[1]
                rv += stride[0]
                i += 1
            layer.delta_biases[f] = np.sum(nx_layer.delta[:, :, f])
        layer.delta = layer.activation_dfn(layer.delta)

class Dropout:
    def __init__(self, prob = 0.5):
        self.input_shape=None
        self.output_shape = None
        self.input_data= None
        self.output = None
        self.isbias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.prob = prob
        self.delta_weights = 0
        self.delta_biases = 0       

    def set_output_shape(self):
        self.output_shape = self.input_shape
        self.weights = 0

    def apply_activation(self, x, train=True):
        if train:
            self.input_data = x
            flat = np.array(self.input_data).flatten()
            random_indices = np.random.randint(0, len(flat), int(self.prob * len(flat)))
            flat[random_indices] = 0
            self.output = flat.reshape(x.shape)
            return self.output
        else:
            self.input_data = x
            self.output = x / self.prob
            return self.output

    def activation_dfn(self, x):
        return x

    def backpropagate(self, nx_layer):
        if type(nx_layer).__name__ != "Conv2d":
            self.error = np.dot(nx_layer.weights, nx_layer.delta)
            self.delta = self.error * self.activation_dfn(self.out)
        else:
            self.delta = nx_layer.delta
        self.delta[self.output == 0] = 0

class Pool2d:
    def __init__(self, kernel_size = (2, 2), stride=None, kind="max", padding=None):
        self.input_shape=None
        self.output_shape = None
        self.input_data= None
        self.output = None
        self.isbias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.delta_weights = 0
        self.delta_biases = 0
        self.padding = padding
        self.p = 1 if padding != None else 0
        self.kernel_size = kernel_size
        if type(stride) == int:
                 stride = (stride, stride)
        self.stride = stride
        if self.stride == None:
            self.stride = self.kernel_size
        self.pools = ['max', "average", 'min']
        if kind not in self.pools:
            raise ValueError("Pool kind not understoood.")            
        self.kind = kind

    def set_output_shape(self):
        self.output_shape = (int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0] + 1), 
                            int((self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1] + 1), self.input_shape[2])

    def apply_activation(self, image):
        stride = self.stride
        kshape = self.kernel_size
        shape = image.shape
        self.input_shape = shape
        self.set_output_shape()
        self.out = np.zeros((self.output_shape))
        for nc in range(shape[2]):
            cimg = []
            rv = 0
            for r in range(kshape[0], shape[0]+1, stride[0]):
                cv = 0
                for c in range(kshape[1], shape[1]+1, stride[1]):
                    chunk = image[rv:r, cv:c, nc]
                    if len(chunk) > 0:                        
                        if self.kind == "max":
                            chunk = np.max(chunk)
                        if self.kind == "min":
                            chunk = np.min(chunk)
                        if self.kind == "average":
                            chunk = np.mean(chunk)
                        cimg.append(chunk)
                    else:
                        cv-=cstep
                    cv+=stride[1]
                rv+=stride[0]
            cimg = np.array(cimg).reshape(int(rv/stride[0]), int(cv/stride[1]))
            self.out[:,:,nc] = cimg
        return self.out

    def backpropagate(self, nx_layer):
        """
        Gradients are passed through index of latest output value .
        """
        layer = self
        stride = layer.stride
        kshape = layer.kernel_size
        image = layer.input
        shape = image.shape
        layer.delta = np.zeros(shape)
        cimg = []
        rstep = stride[0]
        cstep = stride[1]
        for f in range(shape[2]):
            i = 0
            rv = 0
            for r in range(kshape[0], shape[0]+1, rstep):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1]+1, cstep):
                    chunk = image[rv:r, cv:c, f]
                    dout = nx_layer.delta[i, j, f]
                    if layer.kind == "max":
                        p = np.max(chunk)
                        index = np.argwhere(chunk == p)[0]
                        layer.delta[rv+index[0], cv+index[1], f] = dout
                    if layer.kind == "min":
                        p = np.min(chunk)
                        index = np.argwhere(chunk == p)[0]
                        layer.delta[rv+index[0], cv+index[1], f] = dout
                    if layer.kind == "average":
                        p = np.mean(chunk)
                        layer.delta[rv:r, cv:c, f] = dout
                    j+=1
                    cv+=cstep
                rv+=rstep
                i+=1


class Flatten:
    def __init__(self, init_shape = None):
        self.input_shape = None
        self.output_shape = None
        self.input_data = None
        self.output = None
        self.isbias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.delta_weights = 0
        self.delta_biases = 0

    def set_output_shape(self):
        self.output_shape = (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        self.weights = 0

    def apply_activation(self, x):
        self.input_data = x
        self.output = np.array(self.input_data).flatten()
        return self.output

    def activation_dfn(self, x):
        return x

    def backpropagate(self, nx_layer):
        self.error = np.dot(nx_layer.weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfn(self.output)
        self.delta = self.delta.reshape(self.input_shape)
        

class CNN():
    def __init__(self):
        self.layers = []
        self.info_df = {}
        self.column = ["LName", "Input Shape", "Output Shape", "Activation", "Bias"]
        self.parameters = []
        self.optimizer = ""
        self.loss = "mse"
        self.lr = 0.01
        self.mr = 0.0001
        self.metrics = []
        self.av_optimizers = ["sgd", "momentum", "adam"]
        self.av_metrics = ["mse", "accuracy", "cse"]
        self.av_loss = ["mse", "cse"]
        self.iscompiled = False
        self.model_dict = None
        self.out = []
        self.eps = 1e-15
        self.train_loss = {}
        self.val_loss = {}
        self.train_acc = {}
        self.val_acc = {}


    def apply_loss(self, y, out):
        if self.loss == "mse":
            #print(output_shape)
            loss = y - out
            mse = np.mean(np.square(loss), axis=-1)
            return loss, mse

        if self.loss == 'cse':
            """ Requires out to be probability values. """     
            if len(out) == len(y) == 1:
                #print("Using Binary CSE.")
                #y += self.eps
                #out += self.eps
                cse = -(y * np.log(out) + (1 - y) * np.log(1 - out))
                loss = -(y / out - (1 - y) / (1 - out))
                #cse = np.mean(abs(cse))
            else:
                #print("Using Categorical CSE.")
                if self.layers[-1].activation == "softmax":
                    # if o/p layer's fxn is softmax then loss is y - out
                    # check the derivation of softmax and crossentropy with derivative
                    loss = y - out
                    loss = loss / self.layers[-1].activation_dfn(out)
                else:
                    y = np.float64(y)
                    #y += self.eps
                    out += self.eps
                    #cse =  -np.sum(y * (np.log(out)))
                    
                    loss = -(np.nan_to_num(y / out, posinf=0, neginf=0) - np.nan_to_num((1 - y) / (1 - out), posinf=0, neginf=0))
                
            
                cse = -np.sum(y * np.nan_to_num(np.log(out), posinf=0, neginf=0) + (1 - y) * np.nan_to_num(np.log(1 - out), posinf=0, neginf=0))
            return loss, cse


    def add(self, layer):
        if (len(self.layers) > 0):
            prev_layer = self.layers[-1]
            if prev_layer.name != "Input Layer":
                prev_layer.name = f"{type(prev_layer).__name__}{len(self.layers) - 1}"
            if layer.input_shape == None:
                if type(layer).__name__ == "Flatten":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape[:]
                elif type(layer).__name__ == "Conv2d":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape[:] ##???
                elif type(layer).__name__ == "Pool2d":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape[:]
                else:
                    ops = prev_layer.output_shape
                layer.input_shape = ops
                layer.set_output_shape()
            layer.name = f"Out layer({type(layer).__name__})"
        else:
            layer.name = "Input Layer"
        if type(layer).__name__ == "Conv2d":
            if (layer.output_shape[0] <= 0 or layer.output_shape[1] <= 0):
                raise ValueError(f"The output shape became invalid [i.e. {layer.output_shape}]. Reduce filter size or increase image size.")

        self.layers.append(layer)
        self.parameters.append(layer.parameters)

    def summary(self):
        lname = []
        linput = []
        loutput = []
        lactivation = []
        lisbias = []
        lparam = []
        for layer in self.layers:
            lname.append(layer.name)
            linput.append(layer.input_shape)
            loutput.append(layer.output_shape)
            lactivation.append(layer.activation)
            lisbias.append(layer.isbias)
            lparam.append(layer.parameters)
        model_dict = {"Layer Name": lname, "Input": linput, "Output Shape": loutput,
                      "Activation": lactivation, "Bias": lisbias, "Parameters": lparam}
        model_df = pd.DataFrame(model_dict).set_index("Layer Name")
        print(model_df)
        print(f"Total Parameters: {sum(lparam)}")

    def train(self, X, Y, epochs, show_every = 1, batch_size = 32, shuffle = True, val_split = 0.1, val_x = None, val_y = None):
        self.check_trainable(X, Y)
        self.batch_size = batch_size
        t1 = time.time()
        curr_ind = np.arange(0, len(X), dtype=np.int32)
        if shuffle:
            np.random.shuffle(curr_ind)
        if type(val_x) != type(None) and type(val_y) != type(None):
            self.check_trainable(val_x, val_y)
            print("\nValidation data found.\n")
        else:
            val_ex = int(len(X) * val_split)
            val_exs = []
            while len(val_exs) != val_ex:
                rand_ind = np.random.randint(0, len(X))
                if rand_ind not in val_exs:
                    val_exs.append(rand_ind)
            val_ex = np.array(val_exs)
            val_x, val_y = X[val_ex], Y[val_ex]
            curr_ind = np.array([v for v in curr_ind if v not in val_ex])
                             
        print(f"\nTotal {len(X)} samples.\nTraining samples: {len(curr_ind)} Validation samples: {len(val_x)}.")        
        out_activation = self.layers[-1].activation
        batches = []
        len_batch = int(len(curr_ind)/batch_size) 
        if len(curr_ind)%batch_size != 0:
            len_batch+=1
        batches = np.array_split(curr_ind, len_batch)
        
        print(f"Total {len_batch} batches, most batch has {batch_size} samples.\n")
        for e in range(epochs):            
            err = []
            for batch in batches:
                a = [] 
                curr_x, curr_y = X[batch], Y[batch]
                b = 0
                batch_loss = 0
                for x, y in zip(curr_x, curr_y):
                    out = self.feedforward(x)
                    loss, error = self.apply_loss(y, out)
                    #loss = loss.mean(axis=0)
                    batch_loss += loss
                    err.append(error)
                    update = False
                    
                    if b == batch_size-1:
                        update = True
                        loss = batch_loss/batch_size
                    self.backpropagate(loss, update)
                    b+=1
                    
              
            if e % show_every == 0:      
                train_out = self.predict(X[curr_ind])
                train_loss, train_error = self.apply_loss(Y[curr_ind], train_out)
                
                val_out = self.predict(val_x)
                val_loss, val_error = self.apply_loss(val_y, val_out)
                
                if out_activation == "softmax":
                    train_acc = train_out.argmax(axis=1) == Y[curr_ind].argmax(axis=1)
                    val_acc = val_out.argmax(axis=1) == val_y.argmax(axis=1)
                elif out_activation == "sigmoid":
                    train_acc = train_out > 0.7
                    val_acc = val_out > 0.7
                    #pred = pred == Y
                elif out_activation == "linear":
                    train_acc = abs(Y[curr_ind]-train_out) < 0.000001
                    val_acc = abs(Y[val_ex]-val_out) < 0.000001
                    
                self.train_loss[e] = round(train_error.mean(), 4)
                self.train_acc[e] = round(train_acc.mean() * 100, 4)
                
                self.val_loss[e] = round(val_error.mean(), 4)
                self.val_acc[e] = round(val_acc.mean()*100, 4)
                print(f"Epoch: {e}:")
                print(f"Time: {round(time.time() - t1, 3)}sec")
                print(f"Train Loss: {round(train_error.mean(), 4)} Train Accuracy: {round(train_acc.mean() * 100, 4)}%")
                print(f'Val Loss: {(round(val_error.mean(), 4))} Val Accuracy: {round(val_acc.mean() * 100, 4)}% \n')
                
                t1 = time.time()

    def check_trainable(self, X, Y):
    #print(X[0].shape, self.layers[0].input_shape)
        if type(self.layers[0]).__name__ == "Conv2d": 
            if self.iscompiled == False:
                raise ValueError("Model is not compiled.")
            if len(X) != len(Y):
                raise ValueError("Length of training input and label is not equal.")
            if X[0].shape != self.layers[0].input_shape:
                layer = self.layers[0]
                raise ValueError(f"'{layer.name}' expects input of {layer.input_shape} while {X[0].shape[0]} is given.")
            if Y.shape[-1] != self.layers[-1].neurons:
                op_layer = self.layers[-1]
                raise ValueError(f"'{op_layer.name}' expects input of {op_layer.neurons} while {Y.shape[-1]} is given.")  
        else:
            if self.iscompiled == False:
                raise ValueError("Model is not compiled.")
            if len(X) != len(Y):
                raise ValueError("Length of training input and label is not equal.")
            if X[0].shape[0] != self.layers[0].input_shape:
                layer = self.layers[0]
                raise ValueError(f"'{layer.name}' expects input of {layer.input_shape} while {X[0].shape[0]} is given.")
            if Y.shape[-1] != self.layers[-1].neurons:
                op_layer = self.layers[-1]
                raise ValueError(f"'{op_layer.name}' expects input of {op_layer.neurons} while {Y.shape[-1]} is given.")  
        
    def compile_model(self, lr=0.01, mr = 0.001, opt = "sgd", loss = "mse", metrics=['mse']):

        if opt not in self.av_optimizers:
            raise ValueError(f"Optimizer is not understood, use one of {self.av_optimizers}.")
        
        for m in metrics:
            if m not in self.av_metrics:
                raise ValueError(f"Metrics is not understood, use one of {self.av_metrics}.")
        
        if loss not in self.av_loss:
            raise ValueError(f"Loss function is not understood, use one of {self.av_loss}.")
        
        self.optimizer = opt
        self.loss = loss
        self.lr = lr
        self.mr = mr
        self.metrics = metrics
        self.iscompiled = True
        self.optimizer = Optimizer(layers=self.layers, name=opt, learning_rate=lr, mr=mr)
        self.optimizer = self.optimizer.opt_dict[opt]

    def feedforward(self, x, train=True):
        if train:
            for l in self.layers:
                l.input = x            
                x = np.nan_to_num(l.apply_activation(x))
                #print(l.name, x.shape)
                l.out = x

            return x
        else:
            for l in self.layers:
                l.input = x 
                if type(l).__name__ == "Dropout":
                    x = np.nan_to_num(l.apply_activation(x, train=train))
                else:           
                    x = np.nan_to_num(l.apply_activation(x))
                l.out = x

            return x

    def backpropagate(self, loss, update):
        
        # if it is output layer
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:
                if (type(layer).__name__ == "FFL"):
                    layer.error = loss
                    layer.delta = layer.error * layer.activation_dfn(layer.out)
                    layer.delta_weights += layer.delta * np.atleast_2d(layer.input).T
                    layer.delta_biases += layer.delta
            else:
                nx_layer = self.layers[i+1]
                layer.backpropagate(nx_layer)
            if update:
                layer.delta_weights /= self.batch_size
                layer.delta_biases /= self.batch_size
        if update: 
            self.optimizer(self.layers)
            self.zerograd()

    def zerograd(self):
        for l in self.layers:
            try:
                l.delta_weights=np.zeros(l.delta_weights.shape)
                l.delta_biases = np.zeros(l.delta_biases.shape)
            except:
                pass

    def predict(self, X):
        out = []
        if X.shape != self.layers[0].input_shape:
            for x in X:
                out.append(self.feedforward(x, train=False))
            
        else:
            out.append(self.feedforward(X, train = False))
        return np.array(out)

    def accuracy_score(self, y, yt):
        pass
            
    def save_model(self, path="model.json"):
        """
            path:- where to save a model including filename
            saves Json file on given path.
        """

        dict_model = {"model":str(type(self).__name__)}
        to_save = ["name", "isbias", "neurons", "input_shape", "output_shape", 
                    "weights", "biases", "activation", "parameters", "filters",
                    "kernel_size", "padding", "prob", "stride", "kind"]
        for l in self.layers:
            current_layer = vars(l)
            #print(current_layer)
            values = {"type":str(type(l).__name__)}
            for key, value in current_layer.items():
                if key in to_save:
                    if key in ["weights", "biases"]:
                        try:
                            value = value.tolist()
                        except:
                            value = float(value)
                        #print(value)
                    if type(value)== np.int32:
                        value = float(value)
                    if key == "input_shape" or key == "output_shape":
                        try:
                            value = tuple(value)
                        except:
                            pass
                    values[key] = value
            # not all values will be set because many layers can be FFL and this key will be replaced    
            dict_model[l.name] = values
        json_dict = json.dumps(dict_model)    
        with open(path, mode="w") as f:
            f.write(json_dict)
            #print(json_dict)
        print("\nModel Saved.")

    def visualize(self):
        plt.figure(figsize=(10,10))
        k = list(self.train_loss.keys())
        v = list(self.train_loss.values())
        plt.plot(k, v, "g-") 
        k = list(self.val_loss.keys())
        v = list(self.val_loss.values())
        plt.plot(k, v, "r-")
        plt.xlabel("Epochs")
        plt.ylabel(self.loss)
        plt.legend(["Train Loss", "Val Loss"])
        plt.title("Loss vs Epoch")
        plt.show()
        
        plt.figure(figsize=(10,10))
        k = list(self.train_acc.keys())
        v = list(self.train_acc.values())
        plt.plot(k, v, "g-")
        
        k = list(self.val_acc.keys())
        v = list(self.val_acc.values())
        plt.plot(k, v, "r-")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Acc vs epoch")
        plt.legend(["Train Acc", "Val Acc"])
        plt.grid(True)
        plt.show()


class Optimizer:
    """
        A class to perform Optimization of learning parameters.
        Available among ["sgd", "iterative", "momentum", "rmsprop", "adagrad", "adam", "adamax", "adadelta"].
        Object of this class is made inside Compile method of stackking class.

        Example:-
        ----------
        self.optimizer = Optimizer(layers=self.layers, name=opt, learning_rate=lr, mr=mr)
        self.optimizer = self.optimizer.opt_dict[opt]
    """
    def __init__(self, layers, name=None, learning_rate = 0.01, mr=0.001):
        """
        layers:- It is the list of layers on model.
        name:- It is the type of Optimizer.
        learning_rate:- It is the learning rate given by user.
        mr:- It is a momentum rate. Often used on Gradient Descent.
        """
        self.name = name
        self.learning_rate = learning_rate
        self.mr = mr
        keys = ["sgd", "iterative", "momentum", "rmsprop", "adagrad", "adam", "adamax", "adadelta"]
        values = [self.sgd, self.iterative, self.momentum, self.rmsprop, self.adagrad, self.adam, self.adamax, self.adadelta]
        self.opt_dict = {keys[i]:values[i] for i in range(len(keys))}
        if name != None and name in keys:
            self.opt_dict[name](layers=layers, training=False)
            #pass
    def sgd(self, layers, learning_rate=0.01, beta=0.001, training=True):
        learning_rate = self.learning_rate
        for l in layers:
            if l.parameters !=0:
                if training:
                    l.weights += l.pdelta_weights*self.mr + l.delta_weights * learning_rate
                    l.biases += l.pdelta_biases*self.mr + l.delta_biases * learning_rate
                    l.pdelta_weights = l.delta_weights
                    l.pdelta_biases = l.delta_biases
                else:
                    l.pdelta_weights = 0
                    l.pdelta_biases = 0
    def iterative(self, layers, learning_rate=0.01, beta=0, training=True):
        for l in layers:
            if l.parameters !=0:
                l.weights -= learning_rate * l.delta_weights
                l.biases -= learning_rate * l.delta_biases
    def momentum(self, layers, learning_rate=0.1, beta1=0.9, weight_decay=0.0005, nesterov=True, training=True):
        learning_rate = self.learning_rate
        for l in layers:
            if l.parameters !=0:
                if training:
                    l.weights_momentum = beta1 * l.weights_momentum + learning_rate * l.delta_weights-weight_decay *learning_rate*l.weights
                    l.weights+=l.weights_momentum
                    #
                    l.biases_momentum = beta1 * l.biases_momentum + learning_rate * l.delta_biases-weight_decay *learning_rate*l.biases
                    l.biases+=l.biases_momentum
                else:
                    l.weights_momentum = 0
                    l.biases_momentum = 0

            
    def rmsprop(self, layers, learning_rate=0.001, beta1=0.9, epsilon=1e-8, training=True):
        learning_rate=self.learning_rate
        for l in layers:
            if l.parameters !=0:
                if training:
                    l.weights_rms = beta1*l.weights_rms + (1-beta1)*(l.delta_weights ** 2)
                    l.weights += learning_rate * (l.delta_weights/np.sqrt(l.weights_rms + epsilon))
                    l.biases_rms = beta1*l.biases_rms + (1-beta1)*(l.delta_biases ** 2)
                    l.biases += learning_rate * (l.delta_biases/np.sqrt(l.biases_rms + epsilon))
                else:
                    l.weights_rms = 0
                    l.biases_rms = 0
    def adagrad(self, layers, learning_rate=0.01, beta1=0.9, epsilon=1e-8, training=True):
        learning_rate=self.learning_rate
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_adagrad += l.delta_weights ** 2
                    l.weights += learning_rate * (l.delta_weights/np.sqrt(l.weights_adagrad+epsilon))
                    l.biases_adagrad += l.delta_biases ** 2
                    l.biases += learning_rate * (l.delta_biases/np.sqrt(l.biases_adagrad+epsilon))
                else:
                    l.weights_adagrad = 0
                    l.biases_adagrad = 0
    def adam(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0, training=True):
        #print(training)
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.t += 1
                    if l.t == 1:
                        l.pdelta_biases = 0
                        l.pdelta_weights = 0
                    l.weights_adam1 = beta1 * l.weights_adam1 + (1-beta1)*l.delta_weights
                    l.weights_adam2 = beta2 * l.weights_adam2 + (1-beta2)*(l.delta_weights**2)
                    mcap = l.weights_adam1/(1-beta1**l.t)
                    vcap = l.weights_adam2/(1-beta2**l.t)
                    l.delta_weights = mcap/(np.sqrt(vcap) + epsilon)
                    l.weights += l.pdelta_weights * self.mr + learning_rate * l.delta_weights
                    l.pdelta_weights = l.delta_weights * 0

                    l.biases_adam1 = beta1 * l.biases_adam1 + (1-beta1)*l.delta_biases
                    l.biases_adam2 = beta2 * l.biases_adam2 + (1-beta2)*(l.delta_biases**2)
                    mcap = l.biases_adam1/(1-beta1**l.t)
                    vcap = l.biases_adam2/(1-beta2**l.t)
                    l.delta_biases = mcap/(np.sqrt(vcap) +epsilon)
                    l.biases += l.pdelta_biases * self.mr + learning_rate * l.delta_biases
                    l.pdelta_biases = l.delta_biases * 0
                    
                else:
                    l.t = 0
                    l.weights_adam1 = 0
                    l.weights_adam2 = 0
                    l.biases_adam1 = 0
                    l.biases_adam2 = 0
                    
    def adamax(self, layers, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_m = beta1*l.weights_m + (1-beta1)*l.delta_weights
                    l.weights_v = np.maximum(beta2*l.weights_v, abs(l.delta_weights))
                    l.weights += (self.learning_rate/(1-beta1))*(l.weights_m/(l.weights_v+epsilon))
                    
                    l.biases_m = beta1*l.biases_m + (1-beta1)*l.delta_biases
                    l.biases_v = np.maximum(beta2*l.biases_v, abs(l.delta_biases))
                    l.biases += (self.learning_rate/(1-beta1))*(l.biases_m/(l.biases_v+epsilon))
                    
                else:
                    l.weights_m = 0
                    l.biases_m = 0
                    l.weights_v = 0
                    l.biases_v = 0
                    
    def adadelta(self, layers, learning_rate=0.01, beta1=0.9, epsilon=1e-8, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.weights_v = beta1*l.weights_v + (1-beta1)*(l.delta_weights ** 2)
                    l.delta_weights = np.sqrt((l.weights_m+epsilon)/(l.weights_v+epsilon))*l.delta_weights
                    l.weights_m = beta1*l.weights_m + (1-beta1)*(l.delta_weights)
                    l.weights += l.delta_weights
                    
                    l.biases_v = beta1*l.biases_v + (1-beta1)*(l.delta_biases ** 2)
                    l.delta_biases = np.sqrt((l.biases_m+epsilon)/(l.biases_v+epsilon))*l.delta_biases
                    l.biases_m = beta1*l.biases_m+ (1-beta1)*(l.delta_biases)
                    l.biases += l.delta_biases
                    
                else:
                    l.weights_m = 0
                    l.biases_m = 0
                    l.weights_v = 0
                    l.biases_v = 0


def load_model(path="model.json"):
    """
        path:- path of model file including filename        
        returns:- a model
    """
    
    models = {"CNN": CNN}
    #layers = {"FFL": FFL}
    layers = {"FFL": FFL, "Conv2d":Conv2d, "Dropout":Dropout, "Flatten": Flatten, "Pool2d":Pool2d}
    with open(path, "r") as f:
        dict_model = json.load(f)
        model = dict_model["model"]
        #exec("model=models[model]")
        model = models[model]()
        #exec("model=model()")
        for layer, params in dict_model.items():
            if layer != "model":
                # create a layer obj
                lyr_type = layers[params["type"]]
                #print(layers[params["type"]], Conv2d)
                
                ###### create models here.                
                if lyr_type == FFL:                                        
                    lyr.neurons = params["neurons"]
                    lyr = layers[params["type"]](neurons=params["neurons"])
                
                if lyr_type == Conv2d:
                    lyr = layers[params["type"]](filters=int(params["filters"]), kernel_size=params["kernel_size"], padding=params["padding"])
                    #print(params["output_shape"])
                    lyr.out = np.zeros(params["output_shape"])
                    params["input_shape"] = tuple(params["input_shape"])
                    params["output_shape"] = tuple(params["output_shape"])
                if lyr_type == Dropout:
                    lyr = layers[params["type"]](prob=params["prob"])
                    try:
                        params["input_shape"] = tuple(params["input_shape"])
                        params["output_shape"] = tuple(params["output_shape"])
                    except:
                        pass
                    
                if lyr_type == Pool2d:
                    lyr = layers[params["type"]](kernel_size = params["kernel_size"], stride=params["stride"], kind=params["kind"])
                    params["input_shape"] = tuple(params["input_shape"])
                    try:
                        params["output_shape"] = tuple(params["output_shape"])
                    except:
                        pass
                if lyr_type == Flatten:
                    params["input_shape"] = tuple(params["input_shape"])                    
                    lyr = layers[params["type"]](input_shape=params["input_shape"])
                lyr.name = layer
                lyr.activation = params["activation"]
                lyr.isbias = params["isbias"]
                lyr.input_shape = params["input_shape"]
                lyr.output_shape = params["output_shape"]
                lyr.parameters = int(params["parameters"])
                
                if params.get("weights"):
                    lyr.weights = np.array(params["weights"])
                
                if params.get("biases"):
                    lyr.biases = np.array(params["biases"])               
                
                model.layers.append(lyr)
        print("Model Loaded...")
        return model


#######functions
def activate(fxn, r):
    """
    Method to call other activation function.
    """
    activate_dict = {"linear": linear, "tanh": tanh, "sigmoid": sigmoid, "softmax": softmax, "relu": relu}
    return activate_dict[fxn](r)
def deactivate(fxn, r):
    """
    A method to call derivative of activation function.
    """    
    deactivate_dict = {"linear": dlinear, "tanh": dtanh, "sigmoid": dsigmoid, "softmax": dsoftmax, "relu": drelu}
    return deactivate_dict[fxn](r)

def linear(x):
    return x

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    s = np.exp(x)
    return s / np.sum(s)

def relu(x):
    x[x<0] = 0
    return x
    
def dlinear(x):
    return np.ones(x.shape)

def dtanh(x):
    return 2 * x / (1 + x) ** 2 

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)    

def dsoftmax(x):
    soft = softmax(x)
    diag_soft = soft*(1- soft)
    return diag_soft

def drelu(x):
    x[x < 0] = 0
    return x
#######test
'''
img = xt[0]
conv = Conv2d()
conv.input = img
conv.weights = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]).reshape(3, 3, 1, 1)
conv.biases = np.zeros(1)
conv.out = np.zeros((28, 28, 1))
cout = conv.apply_activation(img)
plt.imshow(cout.reshape(28, 28)) 
plt.show()
'''
m = CNN()
m.add(Conv2d(input_shape = (28, 28, 1), filters = 6, padding="same", kernel_size=(5, 5), activation="tanh"))
m.add(Pool2d(kernel_size=(2, 2), stride=2))
m.add(Conv2d(filters=16, kernel_size=(5, 5), padding=None, activation="tanh"))
m.add(Pool2d(kernel_size=(2, 2), stride=2))
m.add(Flatten())
m.add(FFL(neurons = 120, activation = "tanh"))
m.add(FFL(neurons = 100, activation = "tanh"))
m.add(Dropout(0.1))

m.add(FFL(neurons = 10, activation='softmax'))
m.compile_model(lr=0.3, opt="sgd", loss="cse")
m.summary()
m.train(x[:3000], y[:3000], epochs=10, batch_size=30, val_x=xt[:200], val_y=yt[:200])
m.visualize()
m.save_model()
#load_model()
#m.summary()
#print(m.predict(x[10])) 