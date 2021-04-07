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

import numpy as np

# This configuration only has the Convolutional Classifier. 

classifier_config_vision = {

    'tpot.builtins.PytorchConvClassifier': {
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'batch_size': [4, 8, 16, 32],
        'num_epochs': [5, 10, 15],
        'weight_decay': [0, 1e-4, 1e-3, 1e-2],
        'num_conv_layers': [1, 2, 3],
        'num_fc_layers': [1, 2, 3],
        'kernel_proportion_x': [0.25, 0.5],
        'kernel_proportion_y': [0.25, 0.5],
        'featureset_expansion_per_convlayer': [2, 3, 5],
        'feature_reduction_proportion_fclayer': [3, 5, 10],
        'optimizer_name': ['Adam'],
        'activation_func_name': ['ReLU']
    },

    'tpot.builtins.PytorchLSTMClassifier': {
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'batch_size': [4, 8, 16, 32],
        'num_epochs': [5, 10, 15],
        'weight_decay': [0, 1e-4, 1e-3, 1e-2],
        'hidden_size': [1, 2, 3],
        'lstm_layers': [1, 2, 3],
        'optimizer_name': ['Adam'],
        'bidirectionality': [True, False],
        'dropout_perc': [0, 0.05]
    },

}
