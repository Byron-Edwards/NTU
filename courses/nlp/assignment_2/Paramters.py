import torch
from collections import OrderedDict
# parameters for the Model
parameters = OrderedDict()
parameters['train'] = "./data/eng.train"  # Path to train file
parameters['dev'] = "./data/eng.testa"  # Path to test file
parameters['test'] = "./data/eng.testb"  # Path to dev file
parameters['tag_scheme'] = "BIOES"  # BIO or BIOES
parameters['lower'] = True  # Boolean variable to control lowercasing of words
parameters['zeros'] = True  # Boolean variable to control replacement of  all digits by 0
parameters['char_dim'] = 25  # Char embedding dimension
parameters['word_dim'] = 100  # Token embedding dimension
parameters['word_lstm_dim'] = 200  # Token LSTM hidden layer size
parameters['word_bidirect'] = True  # Use a bidirectional LSTM for words
parameters['embedding_path'] = "./data/glove.6B.100d.txt"  # Location of pretrained embeddings
parameters['all_emb'] = 1  # Load all embeddings
parameters['crf'] = 1  # Use CRF (0 to disable)
parameters['dropout'] = 0.5  # Droupout on the input (0 = no dropout)
parameters['epoch'] = 50  # Number of epochs to run"
parameters['weights'] = ""  # path to Pretrained for from a previous run
parameters['name'] = "self-trained"  # Model name
parameters['gradient_clip'] = 5.0
parameters['char_mode'] = "CNN"
parameters['seed'] = 12345

# GPU
parameters['use_gpu'] = torch.cuda.is_available()  # GPU Check
# parameters['reload'] = "./models/pre-trained-model"
parameters['reload'] = False

# Constants
START_TAG = '<START>'
STOP_TAG = '<STOP>'

learning_rate = 0.015
momentum = 0.9
decay_rate = 0.05

# paths to files
# To stored mapping file
mapping_file = './data/mapping.pkl'
use_gpu = parameters['use_gpu']

# To stored model
name = parameters['name']
models_path = "./models/"  # path to saved models
model_name = models_path + name  # get_name(parameters)

