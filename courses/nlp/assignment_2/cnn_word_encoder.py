from __future__ import print_function
import _pickle as cPickle
import time
from datetime import datetime
import urllib
import argparse
import matplotlib.pyplot as plt
import os
import codecs
import numpy as np
from Model import *
from data_prepare import *
from evaluation import *

parser = argparse.ArgumentParser()
parser.add_argument("--multi_cnn", type=int, default=1, help=" use Multi CNN as network")
parser.add_argument("--dilation", type=int, default=0, help="use dilation CNN")
parser.add_argument("--char_mode", type=str, default="CNN", help="char representation")
parser.add_argument("--crf", type=int, default=1, help="Use CRF (0 to disable)")

parser.add_argument("--epochs", default=50, type=int, help="training epoch")
parser.add_argument("--clipgrad", default=5.0, type=float, help="clip of gradient")
parser.add_argument("--lr", default=0.015, type=float, help="SDG learning rate")
parser.add_argument("--momentum", default=0.9, type=float, help="SDG learning rate")
parser.add_argument("--decay_rate", default=0.05, type=float, help="SDG learning rate")
args = parser.parse_args()

parameters['char_mode'] = args.char_mode
parameters['crf'] = args.crf

model_name = "_".join([models_path,
                       datetime.now().strftime("%Y%m%d_%H%M%S"),
                       args.char_mode,
                       str(args.multi_cnn),
                       str(args.dilation),
                       str(args.crf),
                       ])
model_log = model_name + ".log"
logfile = open(model_log, "a")


learning_rate = args.lr
momentum = args.momentum
number_of_epochs = args.epochs
gradient_clip = args.clipgrad
decay_rate = args.decay_rate

plt.rcParams['figure.dpi'] = 80
plt.style.use('seaborn-pastel')
np.random.seed(parameters['seed'])

if not os.path.exists(models_path):
    os.makedirs(models_path)

train_sentences = load_sentences(parameters['train'], parameters['zeros'])
test_sentences = load_sentences(parameters['test'], parameters['zeros'])
dev_sentences = load_sentences(parameters['dev'], parameters['zeros'])

update_tag_scheme(train_sentences, parameters['tag_scheme'])
update_tag_scheme(dev_sentences, parameters['tag_scheme'])
update_tag_scheme(test_sentences, parameters['tag_scheme'])

dico_words, word_to_id, id_to_word = word_mapping(train_sentences, parameters['lower'])
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
)
print("{} / {} / {} sentences in train / dev / test.".format(len(train_data), len(dev_data), len(test_data)))
logfile.write("{} / {} / {} sentences in train / dev / test.\n".format(len(train_data), len(dev_data), len(test_data)))

all_word_embeds = {}
for i, line in enumerate(codecs.open(parameters['embedding_path'], 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

# Intializing Word Embedding Matrix
word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), parameters['word_dim']))

for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))
logfile.write('Loaded %i pretrained embeddings.\n' % len(all_word_embeds))

with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    cPickle.dump(mappings, f)

print('word_to_id: ', len(word_to_id))
logfile.write('word_to_id: {}\n'.format(len(word_to_id)))

# creating the model using the Class defined above
model = BiLSTM_CRF(vocab_size=len(word_to_id),
                   tag_to_ix=tag_to_id,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   char_lstm_dim=parameters['char_dim'],
                   use_gpu=use_gpu,
                   char_to_ix=char_to_id,
                   pre_word_embeds=word_embeds,
                   use_crf=parameters['crf'],
                   char_mode=parameters['char_mode'],
                   multi_cnn=args.multi_cnn,
                   dilation=args.dilation
                   )
print("Model Initialized!!!")


# Reload a saved model, if parameter["reload"] is set to a path
if parameters['reload']:
    if not os.path.exists(parameters['reload']):
        print("downloading pre-trained model")
        model_url = "https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/raw/master/trained-model-cpu"
        urllib.request.urlretrieve(model_url, parameters['reload'])
    model.load_state_dict(torch.load(parameters['reload']))
    print("model reloaded :", parameters['reload'])

if use_gpu:
    model.cuda()

# Initializing the optimizer
# The best results in the paper where achived using stochastic gradient descent (SGD)
# learning rate=0.015 and momentum=0.9
# decay_rate=0.05


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# variables which will used in training process
losses = []  # list to store all losses
loss = 0.0  # Loss Initializatoin
best_dev_F = -1.0  # Current best F-1 Score on Dev Set
best_test_F = -1.0  # Current best F-1 Score on Test Set
best_train_F = -1.0  # Current best F-1 Score on Train Set
all_F = [[0, 0, 0]]  # List storing all the F-1 Scores
eval_every = len(train_data)  # Calculate F-1 Score after this many iterations
plot_every = 2000  # Store loss after this many iterations
count = 0  # Counts the number of iterations

parameters['reload']=False
if not parameters['reload']:
    tr = datetime.now()
    logfile.write("Training Start at {}\n".format(tr.strftime("%Y/%m/%d %H:%M:%S")))
    logfile.write("# Model Parameters: {}\n".format(count_parameters(model)))
    model.train(True)
    for epoch in range(1, number_of_epochs):
        for i, index in enumerate(np.random.permutation(len(train_data))):
            count += 1
            data = train_data[index]
            ##gradient updates for each data entry
            model.zero_grad()

            sentence_in = data['words']

            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']
            chars2 = data['chars']

            if parameters['char_mode'] == 'LSTM':
                chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
                d = {}
                for i, ci in enumerate(chars2):
                    for j, cj in enumerate(chars2_sorted):
                        if ci == cj and not j in d and not i in d.values():
                            d[j] = i
                            continue
                chars2_length = [len(c) for c in chars2_sorted]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
                for i, c in enumerate(chars2_sorted):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            if parameters['char_mode'] == 'CNN':

                d = {}

                ## Padding the each word to max word size of that sentence
                chars2_length = [len(c) for c in chars2]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                for i, c in enumerate(chars2):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            targets = torch.LongTensor(tags)

            # we calculate the negative log-likelihood for the predicted tags using the predefined function
            if use_gpu:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(),
                                                              chars2_length, d)
            else:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length, d)
            loss += neg_log_likelihood.data.item() / len(data['words'])
            neg_log_likelihood.backward()

            # we use gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clip)
            optimizer.step()

            # Storing loss
            if count % plot_every == 0:
                loss /= plot_every
                print(count, ': ', loss)
                if losses == []:
                    losses.append(loss)
                losses.append(loss)
                loss = 0.0

            # Evaluating on Train, Test, Dev Sets
            if count % (eval_every) == 0 and count > (eval_every * 20) or \
                    count % (eval_every * 4) == 0 and count < (eval_every * 20):
                model.train(False)
                best_train_F, new_train_F, _ = evaluating(model, train_data, tag_to_id, best_train_F, "Train")
                logfile.write("{}: new_F: {} best_F: {} \n".format( "Train", new_train_F, best_train_F))

                best_dev_F, new_dev_F, save = evaluating(model, dev_data, tag_to_id, best_dev_F, "Dev")
                logfile.write("{}: new_F: {} best_F: {} \n".format("Dev", new_dev_F, best_dev_F))

                best_test_F, new_test_F, _ = evaluating(model, test_data, tag_to_id, best_test_F, "Test")
                logfile.write("{}: new_F: {} best_F: {} \n".format("Test", new_test_F, best_test_F))

                if save:
                    print("Saving Model to ", model_name)
                    torch.save(model.state_dict(), model_name)
                    logfile.write("Saving Model to {} at {}\n".format(model_name,datetime.now().strftime("%Y/%m/%d %H:%M:%S")))

                all_F.append([new_train_F, new_dev_F, new_test_F])
                model.train(True)

            # Performing decay on the learning rate
            if count % len(train_data) == 0:
                adjust_learning_rate(optimizer, lr=learning_rate / (1 + decay_rate * count / len(train_data)))

    logfile.write("Training End at {}\n".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    print((datetime.now() - tr).seconds)
    logfile.write("Training Time {}s \n".format((datetime.now() - tr).seconds))
    logfile.close()
    # plt.plot(losses)
    # plt.show()
