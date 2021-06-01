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

# This configuration only has full-network NN classifiers for sequence/text data. 

classifier_config_sequence = {

    'tpot.builtins.PytorchLSTMClassifier': {
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'batch_size': [4, 8, 16, 32],
        'num_epochs': [5, 10, 15],
        'weight_decay': [0, 1e-4, 1e-3, 1e-2],
        'hidden_size': [1, 2, 3, 5, 10],
        'lstm_layers': [1, 2, 3, 5, 10],
        'optimizer_name': ['Adam', 'SGD', 'RMSprop'],
        'bidirectionality': [True, False],
        'dropout_prop': [0, 0.1, 0.2, 0.4]
    },

}
