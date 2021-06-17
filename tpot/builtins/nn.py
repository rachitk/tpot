# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

# Note: There are quite a few pylint messages disabled in this file. In
# general, this usually should be avoided. However, in some cases it is
# necessary: e.g., we use `X` and `y` to refer to data and labels in compliance
# with the scikit-learn API, but pylint doesn't like short variable names.

# pylint: disable=redefined-outer-name
# pylint: disable=not-callable

from abc import abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, assert_all_finite, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target

#Needed for torch
try:
    import torch
    from torch import nn
    from torch.autograd import Variable
    from torch.optim import Adam
    from torch.utils.data import TensorDataset, DataLoader, Dataset
except ModuleNotFoundError:
    raise

#Attempt to import torchvision for the prebuilt networks, but also just pass if not found
#Error handling for torchvision import failure is done in the Pytorch prebuilt CONV classifier.
try:
    import torchvision
    from torchvision import models, transforms
except:
    pass

def _pytorch_model_is_fully_initialized(clf: BaseEstimator):
    if all([
        hasattr(clf, 'network'),
        hasattr(clf, 'loss_function'),
        hasattr(clf, 'optimizer'),
        hasattr(clf, 'data_loader'),
        hasattr(clf, 'train_dset_len'),
        hasattr(clf, 'device')
    ]):
        return True
    else:
        return False

def _get_cuda_device_if_available():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def _create_optimizer(optimizer_name, **kwargs):
    #Only supports optimizers in PyTorch that accept the arguments passed in (lr/learning rate, weight_decay)

    try:
        optim_func = getattr(torch.optim, optimizer_name)
    except AttributeError:
        raise NotImplementedError("{} is not a valid PyTorch optimizer".format(optimizer_name))

    optim_def = optim_func(**kwargs)

    return optim_def


def _create_activation_func(activation_func_name):
    #Only supports activation functions in PyTorch that do not require arguments (ReLU, Tanh, etc.)
    #Optimizing activation function parameters would explode tree complexity too much
    #All optimizers can be found in Pytorch's nn documentation

    try:
        activ_func = getattr(nn, activation_func_name)
    except AttributeError:
        raise NotImplementedError("{} is not a valid PyTorch NN activation function".format(activation_func_name))

    return activ_func()


class PytorchEstimator(BaseEstimator):
    """Base class for Pytorch-based estimators (currently only classifiers) for
    use in TPOT.

    In the future, these will be merged into TPOT's main code base.
    """

    @abstractmethod
    def fit(self, X, y): # pragma: no cover
        pass

    @abstractmethod
    def transform(self, X): # pragma: no cover
        pass

    def predict(self, X):
        return self.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class PytorchClassifier(PytorchEstimator, ClassifierMixin):
    @abstractmethod
    def _init_model(self, X, y): # pragma: no cover
        pass

    def fit(self, X, y):
        """Generalizable method for fitting a PyTorch estimator to a training
        set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        # pylint: disable=no-member

        self._init_model(X, y)

        assert _pytorch_model_is_fully_initialized(self)

        for epoch in range(self.num_epochs):
            for i, (samples, labels) in enumerate(self.data_loader):
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.network(samples)

                try:
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                except Exception as e:
                    print('Something weird happened with the loss function (crashed)')
                    print(outputs)
                    print(labels)
                    raise

                self.optimizer.step()

                if self.verbose:
                    logging_step = max(min((self.train_dset_len // self.batch_size) // 10, 100),1)
                    print('Logging every {} step(s)'.format(logging_step))

                    if ((i + 1) % logging_step == 0):
                        print(
                            "Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f"
                            % (
                                epoch + 1,
                                self.num_epochs,
                                i + 1,
                                np.ceil(self.train_dset_len / self.batch_size),
                                loss.item(),
                            )
                        )

                    if(np.isnan(loss.item())):
                        print('Loss function produced NaN...')
                        print(outputs)
                        print(labels)

        # pylint: disable=attribute-defined-outside-init
        self.is_fitted_ = True
        return self

    def validate_inputs(self, X, y):
        # Things we don't want to allow until we've tested them:
        # - Sparse inputs
        # - Multiclass outputs (e.g., more than 2 classes in `y`)
        # - Non-finite inputs
        # - Complex inputs

        # Check to see if the Pytorch estimator has the n-d attribute set
        # In future, should probably just explicitly define this for all estimators that inherit this class
        if hasattr(self, "allow_nd"):
            X, y = check_X_y(X, y, accept_sparse=False, allow_nd=self.allow_nd)
        else:
            X, y = check_X_y(X, y, accept_sparse=False, allow_nd=False)


        assert_all_finite(X, y)

        #if type_of_target(y) != 'binary':
        #    raise ValueError("Non-binary targets not supported")

        if np.any(np.iscomplex(X)) or np.any(np.iscomplex(y)):
            raise ValueError("Complex data not supported")
        if np.issubdtype(X.dtype, np.object_) or np.issubdtype(y.dtype, np.object_):
            try:
                X = X.astype(float)
                y = y.astype(int)
            except (TypeError, ValueError):
                raise ValueError("argument must be a string.* number")

        return (X, y)

    def predict(self, X):
        # pylint: disable=no-member

        if hasattr(self, "allow_nd"):
            X = check_array(X, accept_sparse=True, allow_nd=self.allow_nd)
        else:
            X = check_array(X, accept_sparse=True, allow_nd=False)

        check_is_fitted(self, 'is_fitted_')

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        predictions = np.empty(len(X), dtype=int)
        for i, rows in enumerate(X):
            rows = Variable(rows.view(-1, self.input_size))
            outputs = self.network(rows)

            _, predicted = torch.max(outputs.data, 1)
            predictions[i] = int(predicted)
        return predictions.reshape(-1, 1)

    def transform(self, X):
        return self.predict(X)



class _LR(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, input_size, num_classes):
        super(_LR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

class _MLP(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, input_size, num_classes):
        super(_MLP, self).__init__()

        self.hidden_size = round((input_size+num_classes)/2)

        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        hidden = self.fc1(x)
        r1 = self.relu(hidden)
        out = self.fc2(r1)
        return out

class _CONV(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, input_size, num_classes, num_conv_layers, num_fc_layers, 
        kernel_proportion_x, kernel_proportion_y,
        featureset_expansion_per_convlayer, feature_reduction_proportion_fclayer,
        activation_func_name):

        super(_CONV, self).__init__()

        #Determine activation function by names given
        activ_func = _create_activation_func(activation_func_name)

        #Note that input size is a tuple: (N, C, H, W) with C being the number of channels
        #Determine kernel sizes for each of the convolutional layers
        #Forces the minimum size of a kernel to be 1 in either dimension
        #Checking to ensure that the size of the output of each layer never becomes 1x1 or smaller
        #Size calculated using [(W-K+2P)/S + 1], with P=0 and S=1 (no padding, stride=1)
        #Out_size in format [H, W, C]
        #All convolutional layers will have the activation function following it to introduce nonlinearity
        #If a kernel is 1x1 then no need to continue convolutions

        #Min kernel is 2x2 unless the original input size is 1 in either dimension, if so that dim is forced to 1

        k_sizes = np.array([min(input_size[2], max(2., np.ceil(input_size[2]*kernel_proportion_x))), 
            min(input_size[3], max(2., np.ceil(input_size[3]*kernel_proportion_y)))])
        out_sizes = np.array([input_size[2]-k_sizes[0]+1, input_size[3]-k_sizes[1]+1, 
            input_size[1]*featureset_expansion_per_convlayer])

        self.conv_layers = nn.ModuleList()
        conv1 = nn.Conv2d(in_channels=int(input_size[1]), out_channels=int(out_sizes[2]), 
            kernel_size=(int(k_sizes[0]), int(k_sizes[1])))
        self.conv_layers.append(conv1)
        self.conv_layers.append(activ_func)

        conv_layers_used = 1

        for i in range(1, num_conv_layers):
            #k_sizes and out_sizes are not lists of lists on the first iteration of the loop, so can't subindex
            
            #Min kernel is 2x2 unless the original input size is 1 in either dimension, if so that dim is forced to 1
            if(conv_layers_used == 1):
                next_ksizes = np.array([min(input_size[2], max(2., np.ceil(out_sizes[0]*kernel_proportion_x))), 
                    min(input_size[2], max(2., np.ceil(out_sizes[1]*kernel_proportion_y)))])

                next_outsizes = np.array([out_sizes[0]-next_ksizes[0]+1, out_sizes[1]-next_ksizes[1]+1, 
                    out_sizes[2]*featureset_expansion_per_convlayer])

            else:
                next_ksizes = np.array([min(input_size[2], max(2., np.ceil(out_sizes[i-1][0]*kernel_proportion_x))), 
                    min(input_size[2], max(2., np.ceil(out_sizes[i-1][1]*kernel_proportion_y)))])

                next_outsizes = np.array([out_sizes[i-1][0]-next_ksizes[0]+1, out_sizes[i-1][1]-next_ksizes[1]+1, 
                    out_sizes[i-1][2]*featureset_expansion_per_convlayer])

            #stop creating layers if either output dim would be < 1, or if the kernel == [1,1]
            if(next_outsizes[0] < 1 or next_outsizes[1] < 1 or np.array_equal(next_ksizes, np.array([1,1]))):
                break
            else:
                conv_layers_used += 1
                k_sizes = np.vstack((k_sizes, next_ksizes))
                out_sizes = np.vstack((out_sizes, next_outsizes))
                
                conv_next = nn.Conv2d(in_channels=int(out_sizes[i-1][2]), out_channels=int(next_outsizes[2]), 
                    kernel_size=(int(next_ksizes[0]), int(next_ksizes[1])))
                self.conv_layers.append(conv_next)
                self.conv_layers.append(activ_func)

                #Cease adding layers if the current image output is 1,1 since there's nothing left to convolve
                if(next_outsizes == [1,1]):
                    break


        #Construct fully connected layers using the final output sizes of the network and going down
        #Using proportion of feature_reduction_proportion_fclayer to determine how many features each FC outputs
        #Final FC layer will need to be the number of classes

        #Check if only one convolutional layer was used 
        #(in which case out_sizes is just a single list, not a list of lists)
        if(conv_layers_used == 1):
            conv_out_features = int(np.prod(out_sizes))
        else:
            conv_out_features = int(np.prod(out_sizes[-1]))

        #For use when flattening later
        self.conv_out_features = conv_out_features

        fc_featurenums = [int(np.ceil(conv_out_features//feature_reduction_proportion_fclayer))]

        self.fc_layers = nn.ModuleList()

        #If just one fc layer or if the feature reduction proportion implies only one layer needed, 
        #then the final layer will reduce to the number of classes (rather than needing to upscale)
        #If more than one, need to scale out how the features are reduced across layers
        if(num_fc_layers == 1 or fc_featurenums[0] <= num_classes):
            fc1 = nn.Linear(conv_out_features, num_classes)
            self.fc_layers.append(fc1)
        else:
            fc1 = nn.Linear(conv_out_features, fc_featurenums[0])
            self.fc_layers.append(fc1)
            self.fc_layers.append(activ_func)

            for j in range(num_fc_layers-1):

                next_featurenums = int(np.ceil(fc_featurenums[j]//feature_reduction_proportion_fclayer))

                if(next_featurenums <= num_classes):
                    break
                else:
                    fc_featurenums.append(next_featurenums)

                    fcnext = nn.Linear(fc_featurenums[j], next_featurenums)
                    self.fc_layers.append(fcnext)
                    self.fc_layers.append(activ_func)

            fc_final = nn.Linear(fc_featurenums[-1], num_classes)
            self.fc_layers.append(fc_final)


    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)

        x = x.view(-1, self.conv_out_features)

        for i, layer in enumerate(self.fc_layers):
            x = layer(x)

        return x



class _pytorchPrebuiltCONV(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, num_classes, pretrained_val, network_name):

        super(_pytorchPrebuiltCONV, self).__init__()

        network_name = network_name.lower()

        #Note that input size is: (N, 3, 224, 224) to match every prebuilt network.

        #Every network that is supported will be grabbed from the Pytorch model zoo (trained on imagenet). 
        #Need to change final layer to output the correct number of classes for the paradigm.
        #Changing the final layer will have to be different for every single network because they aren't constructed the same way.
        #As such, support for other networks will need to be manually added. This can't be automated like the Optimizer or Activation Function dynamic assignments.

        #Currently accepts any of the following as input for network name:  
        #[resnet, alexnet, vgg, squeezenet, densenet, googlenet, shufflenet, mobilenet, resnext, wide_resnet, mnasnet]
        #(Not including inception because it has a different input size of 299x299 instead)


        if(network_name == 'resnet'):
            model = models.resnet18(pretrained=pretrained_val)
            model.fc = nn.Linear(512, num_classes)

        elif(network_name == 'alexnet'):
            model = models.alexnet(pretrained=pretrained_val)
            model.classifier[6] = nn.Linear(4096, num_classes)

        elif(network_name == 'vgg'):
            model = models.vgg16(pretrained=pretrained_val)
            model.classifier[6] = nn.Linear(4096, num_classes)

        elif(network_name == 'squeezenet'):
            model = models.squeezenet1_0(pretrained=pretrained_val)
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

        elif(network_name == 'densenet'):
            model = models.densenet161(pretrained=pretrained_val)
            model.classifier = nn.Linear(1024, num_classes)

        # elif(network_name == 'inception'):
        #     model = models.inception_v3(pretrained=pretrained_val)

        elif(network_name == 'googlenet'):
            model = models.googlenet(pretrained=pretrained_val)
            model.fc = nn.Linear(1024, num_classes)

        elif(network_name == 'shufflenet'):
            model = models.shufflenet_v2_x1_0(pretrained=pretrained_val)
            model.fc = nn.Linear(1024, num_classes)

        elif(network_name == 'mobilenet'):
            model = models.mobilenet_v2(pretrained=pretrained_val)
            model.classifier[1] = nn.Linear(1280, num_classes)

        elif(network_name == 'resnext'):
            model = models.resnext50_32x4d(pretrained=pretrained_val)
            model.fc = nn.Linear(2048, num_classes)

        elif(network_name == 'wide_resnet'):
            model = models.wide_resnet50_2(pretrained=pretrained_val)
            model.fc = nn.Linear(2048, num_classes)

        elif(network_name == 'mnasnet'):
            model = models.mnasnet1_0(pretrained=pretrained_val)
            model.classifier[1] = nn.Linear(1280, num_classes)

        else:
            raise NotImplementedError("{} is not supported in TPOT yet. \
                Supported: [resnet, alexnet, vgg, squeezenet, densenet, googlenet, shufflenet, mobilenet, resnext, wide_resnet, mnasnet]".format(network_name))


        self.nn_model = model


    def forward(self, x):
        x = self.nn_model(x)

        return x


class _LSTM(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, in_features, num_classes, hidden_size, num_layers, bidirectionality, dropout_prop, need_embeddings, vocab_size):
        super(_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectionality = bidirectionality
        self.need_embeddings = need_embeddings

        #Expects that any padding will have been done with index 0
        if(need_embeddings):
            self.embedding_layer = nn.Embedding(vocab_size, in_features, padding_idx=0)

        #LSTM definition (batch first for consistency with all other types of layers)
        self.lstm_layer = nn.LSTM(in_features, hidden_size, num_layers=num_layers, dropout=dropout_prop, batch_first=True, bidirectional=bidirectionality)

        #Final fully connected to output to number of classes (multiply by two if bidirectional)
        if(bidirectionality):
            self.fc = nn.Linear(hidden_size*2*num_layers, num_classes)
        else:
            self.fc = nn.Linear(hidden_size*num_layers, num_classes)

    def forward(self, x):
        num_directions = 2 if self.bidirectionality else 1

        #Create embeddings if needed
        if(self.need_embeddings):
            x = self.embedding_layer(x)

        #init hidden
        h_0, c_0 = self.init_hidden(x, num_directions)
        #Feed sequences through the LSTM layer
        lstm_output, (h_f, c_f) = self.lstm_layer(x, (h_0, c_0))
        #Reshape the final hidden output to a shape [batch_size, hidden_size * num_layers * num_directions]
        h_f = h_f.view(-1, self.hidden_size * self.num_layers * num_directions)
        #Output final connected layer (returns shape: [batch_size, num_classes])
        out = self.fc(h_f)

        return out

    def init_hidden(self, x, num_directions):
        batch_size = x.shape[0]

        h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size)

        return [n for n in (h_0, c_0)]



class PytorchLRClassifier(PytorchClassifier):
    """Logistic Regression classifier, implemented in PyTorch, for use with
    TPOT.

    For examples on standalone use (i.e., non-TPOT) refer to:
    https://github.com/trang1618/tpot-nn/blob/master/tpot_nn/estimator_sandbox.py
    """

    def __init__(
        self,
        num_epochs=10,
        batch_size=16,
        learning_rate=0.02,
        weight_decay=1e-4,
        verbose=False
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.input_size = None
        self.num_classes = None
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.data_loader = None
        self.train_dset_len = None
        self.device = None

    def _init_model(self, X, y):
        device = _get_cuda_device_if_available()

        X, y = self.validate_inputs(X, y)

        self.input_size = X.shape[-1]
        self.num_classes = len(set(y))

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        train_dset = TensorDataset(X, y)

        # Set parameters of the network
        self.network = _LR(self.input_size, self.num_classes).to(device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.data_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.train_dset_len = len(train_dset)
        self.device = device

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': True}

class PytorchMLPClassifier(PytorchClassifier):
    """Multilayer Perceptron, implemented in PyTorch, for use with TPOT.
    """

    def __init__(
        self,
        num_epochs=10,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0,
        verbose=False
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.input_size = None
        self.num_classes = None
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.data_loader = None
        self.train_dset_len = None
        self.device = None

    def _init_model(self, X, y):
        device = _get_cuda_device_if_available()

        X, y = self.validate_inputs(X, y)

        self.input_size = X.shape[-1]
        self.num_classes = len(set(y))

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        train_dset = TensorDataset(X, y)

        # Set parameters of the network
        self.network = _MLP(self.input_size, self.num_classes).to(device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.data_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.train_dset_len = len(train_dset)
        self.device = device

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': True}

class PytorchConvClassifier(PytorchClassifier):
    """Convolutional layer classifier, implemented in PyTorch, for use with TPOT.
    """

    def __init__(
        self,
        num_epochs=10,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0,
        verbose=False,
        num_conv_layers=1,
        num_fc_layers=1,
        kernel_proportion_x=0.05,
        kernel_proportion_y=0.05,
        featureset_expansion_per_convlayer=3,
        feature_reduction_proportion_fclayer=10,
        optimizer_name="Adam",
        activation_func_name="ReLU"
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.num_conv_layers=num_conv_layers
        self.num_fc_layers=num_fc_layers
        self.kernel_proportion_x=kernel_proportion_x
        self.kernel_proportion_y=kernel_proportion_y
        self.featureset_expansion_per_convlayer=featureset_expansion_per_convlayer
        self.feature_reduction_proportion_fclayer=feature_reduction_proportion_fclayer
        self.optimizer_name=optimizer_name
        self.activation_func_name=activation_func_name

        self.input_size = None
        self.num_classes = None
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.data_loader = None
        self.train_dset_len = None
        self.device = None

        #Unique classifier that allows for N-D inputs (assumed to be images)
        self.allow_nd = True

    def _init_model(self, X, y):
        device = _get_cuda_device_if_available()

        X, y = self.validate_inputs(X, y)

        init_input_size = X.shape

        # Place X into the expected form if only 1 channel input but as a 3D array
        # (as expected size is 4D with [N, 1, H, W])
        if(X.ndim == 3):
            X = X.reshape(init_input_size[0], -1, init_input_size[1], init_input_size[2])

        self.input_size = X.shape
        self.num_classes = len(set(y))

        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        train_dset = TensorDataset(X, y)

        # Set parameters of the network
        self.network = _CONV(
            self.input_size, self.num_classes, self.num_conv_layers, 
            self.num_fc_layers, self.kernel_proportion_x, self.kernel_proportion_y, self.featureset_expansion_per_convlayer,
            self.feature_reduction_proportion_fclayer, self.activation_func_name
        ).to(device)
        
        self.loss_function = nn.CrossEntropyLoss()
        #self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer = _create_optimizer(self.optimizer_name, params=self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.data_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.train_dset_len = len(train_dset)
        self.device = device

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': False}

    def predict(self, X):
        """Special predict method for convolutional implementations
        Will make super class handle this properly in the future
        """

        if hasattr(self, "allow_nd"):
            X = check_array(X, accept_sparse=True, allow_nd=self.allow_nd)
        else:
            X = check_array(X, accept_sparse=True, allow_nd=False)

        X_size = X.shape

        if(X.ndim == 3):
            X = X.reshape(X_size[0], -1, X_size[1], X_size[2])

        X_size_4D = X.shape

        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        predictions = np.empty(X_size_4D[0], dtype=int)

        #Feed images into the network (in the appropriate size for the network)
        #Then store only the most highly predicted class for each
        outputs = self.network(X)
        _, predicted = torch.max(outputs.data, 1)
        predictions = predicted.tolist()
        return np.reshape(predictions, (-1, 1))



class PrebuiltPytorchConvClassifier(PytorchClassifier):
    """Convolutional layer classifier, making use of prebuilt architectures in PyTorch, for use with TPOT.
    """

    def __init__(
        self,
        num_epochs=10,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0,
        verbose=True,
        network_name="AlexNet",
        pretrained=True,
        optimizer_name="Adam",
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.network_name = network_name
        self.pretrained = pretrained
        self.optimizer_name=optimizer_name

        self.input_size = None
        self.num_classes = None
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.data_loader = None
        self.train_dset_len = None
        self.device = None

        #Unique classifier that allows for N-D inputs (assumed to be images)
        self.allow_nd = True

    def _init_model(self, X, y):
        device = _get_cuda_device_if_available()

        X, y = self.validate_inputs(X, y)

        self.num_classes = len(set(y))


        # Place X into the expected form if only 1 channel input but as a 3D array (aka [N, H, W])
        # (as expected size for all prebuilt models is 4D with [N, 3, H, W], need to stack to RGB 
        # if X is only a sequence of 2D images)
        init_input_size = X.shape
        if(X.ndim == 3):
            X = np.stack((X, X, X), axis=-1)
            X = X.reshape(init_input_size[0], -1, init_input_size[1], init_input_size[2])

        # X = torch.tensor(X, dtype=torch.float32).to(device)
        # y = torch.tensor(y, dtype=torch.long).to(device)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        #Create normalizing transform for all prebuilt models in torchvision (except inception, which is unsupported)
        #Also running the check here to see if torchvision is installed; if not, raise an error
        try:
            resize_norm_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
             )])
        except ModuleNotFoundError:
            raise

        #Apply normalizing transform - now done with dataloader
        # X = [resize_norm_transform(im) for im in X]
        # X = torch.stack(X)

        self.input_size = X.shape


        train_dset = CustomTensorDataset(X, y, resize_norm_transform)

        # Set parameters of the network
        self.network = _pytorchPrebuiltCONV(
            self.num_classes, self.pretrained, self.network_name
        ).to(device)
        
        self.loss_function = nn.CrossEntropyLoss()
        #self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer = _create_optimizer(self.optimizer_name, params=self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.data_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.train_dset_len = len(train_dset)
        self.device = device

        if(self.verbose):
            print("Network: {}; optimizer: {}; epochs: {}; batch size: {}; num samples: {}; pretrained: {}".
                format(self.network_name, self.optimizer_name, self.num_epochs, self.batch_size, y.shape, self.pretrained))


    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': False}

    def predict(self, X):
        """Special predict method for prebuilt convolutional implementations
        """

        if hasattr(self, "allow_nd"):
            X = check_array(X, accept_sparse=True, allow_nd=self.allow_nd)
        else:
            X = check_array(X, accept_sparse=True, allow_nd=False)

        # Place X into the expected form if only 1 channel input but as a 3D array
        # (as expected size for all prebuilt models is 4D with [N, 3, H, W], need to stack to RGB 
        # if X is only a sequence of 2D images)
        init_input_size = X.shape
        if(X.ndim == 3):
            X = np.stack((X, X, X), axis=-1)
            X = X.reshape(init_input_size[0], -1, init_input_size[1], init_input_size[2])

        X = torch.tensor(X, dtype=torch.float32)

        #Create normalizing transform for all prebuilt models in torchvision (except inception, which is unsupported)
        #Also running the check here to see if torchvision is installed; if not, raise an error
        try:
            resize_norm_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
             )])
        except ModuleNotFoundError:
            raise

        #Apply normalizing transform
        # X = [resize_norm_transform(im) for im in X]
        # X = torch.stack(X)


        predictions = np.empty(init_input_size[0], dtype=int)

        #Feed images into the network (in the appropriate size for the network)
        #Then store only the most highly predicted class for each
        #NOTE: uses a lot of memory as the transforms need to be applied to all the images at once

        # outputs = self.network(X)
        # _, predicted = torch.max(outputs.data, 1)
        # predictions = predicted.tolist()
        # return np.reshape(predictions, (-1, 1))

        #To try to avoid hitting memory limits, only feed in images in one at a time 
        #(and unsqueeze to make it 4D)
        #Applying the transform to each as it is passed in
        for i, im in enumerate(X):
            transformedIm = torch.unsqueeze(resize_norm_transform(im),0).to(self.device)

            output = self.network(transformedIm)
            _, prediction = torch.max(output.data, 1)

            predictions[i] = int(prediction)

        return np.reshape(predictions, (-1, 1))




class PytorchLSTMClassifier(PytorchClassifier):
    """LSTM (an RNN) classifier, implemented in PyTorch, for use with TPOT.
    """

    def __init__(
        self,
        num_epochs=10,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0,
        verbose=False,
        hidden_size=1,
        lstm_layers=1,
        optimizer_name="Adam",
        bidirectionality=False,
        dropout_prop=0
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.hidden_size=hidden_size
        self.lstm_layers=lstm_layers
        self.optimizer_name=optimizer_name
        self.bidirectionality=bidirectionality
        self.dropout_prop=dropout_prop

        self.input_size = None
        self.num_classes = None
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.data_loader = None
        self.train_dset_len = None
        self.device = None

        self.vocab_size = None
        self.need_embeddings = None

        #Unique classifier that allows for N-D inputs (assumed to be sequences or values to be encoded)
        self.allow_nd = True

    def _init_model(self, X, y):
        device = _get_cuda_device_if_available()

        X, y = self.validate_inputs(X, y)

        #Expected shape to be 3D in the format (batch size, sequence length, features) if pre-encoded or standard sequence data
        #If passed in as only 2D with the encoded word indexes for each sequence, will need an embedding layer
        #and so will define input features as the embedding_dim (10) and the vocab_size as the max of the input data
        self.input_size = X.shape

        if(len(self.input_size) == 2):
            self.vocab_size = np.max(X)+1
            #maybe find a way to make features modifiable? Another network entirely that allows user to predecide if encodings needed?
            self.input_features = 10 
            self.need_embeddings = True
        elif(len(self.input_size) == 3):
            self.input_features = self.input_size[-1]
            self.need_embeddings = False

        self.num_classes = len(set(y))

        #Needs long for X (for embeddings layer)
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        train_dset = TensorDataset(X, y)

        # Set parameters of the network
        self.network = _LSTM(
            self.input_features, self.num_classes, self.hidden_size, 
            self.lstm_layers, self.bidirectionality, self.dropout_prop,
            self.need_embeddings, self.vocab_size
        ).to(device)
        
        self.loss_function = nn.CrossEntropyLoss()
        #self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer = _create_optimizer(self.optimizer_name, params=self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.data_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.train_dset_len = len(train_dset)
        self.device = device

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': False}

    def predict(self, X):
        """Special predict method for LSTM implementations
        """

        if hasattr(self, "allow_nd"):
            X = check_array(X, accept_sparse=True, allow_nd=self.allow_nd)
        else:
            X = check_array(X, accept_sparse=True, allow_nd=False)

        X_size = X.shape

        X = torch.tensor(X, dtype=torch.long).to(self.device)

        predictions = np.empty(X_size[0], dtype=int)

        #Check what the input dimension is and feed in appropriately
        #Feed each sequence into the network (in the appropriate size for the network)
        #Then store only the most highly predicted class
        if(self.need_embeddings):
            #Replace all indices that didn't appear in the original training set with 1s (unknowns)
            max_vocab_size = self.vocab_size-1
            X[X>max_vocab_size] = 1
            #X = torch.where(X>max_vocab_size, 1, X)  #Same as above, but less efficient (apparently?)

            outputs = self.network(X)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.tolist()
            return np.reshape(predictions, (-1, 1))

        else:
            outputs = self.network(X)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.tolist()
            return np.reshape(predictions, (-1, 1))



class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms. Done to improve memory usage.
    """
    def __init__(self, X, y, transform=None):
        self.X_vals = X
        self.y_vals = y
        self.transform = transform

    def __getitem__(self, index):
        x_out = self.X_vals[index]

        if self.transform:
            x_out = self.transform(x_out)

        y_out = self.y_vals[index]

        return x_out, y_out

    def __len__(self):
        return self.X_vals.size(0)

