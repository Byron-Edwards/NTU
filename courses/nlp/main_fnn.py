# coding: utf-8
import argparse
import time
from datetime import datetime
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='FNN',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--ngrams', type=int, default=6,
                    help='number of previous word for training')
# parser.add_argument('--nlayers', type=int, default=2,
#                     help='number of layers')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
# parser.add_argument('--clip', type=float, default=0.25,
#                     help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1000, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=1,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

# parser.add_argument('--nhead', type=int, default=2,
#                     help='the number of heads in the encoder/decoder of the transformer model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    # data = data.view(bsz, -1).t().contiguous()
    data = data.view(nbatch, -1).contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.FNNModel(ntokens, args.ngrams,args.emsize, args.nhid, args.dropout, tie_weights=args.tied).to(device)
model = nn.DataParallel(model).to(device)
model = model.module
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
###############################################################################
# Training code
###############################################################################

# def get_batch(source, i):
#     seq_len = min(args.bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target

def get_batch(source,i,steps=1):
    batch_data = source[i]
    data = batch_data[0:-1].unfold(0,args.ngrams,steps)
    target = batch_data[args.ngrams:].unfold(0,1,steps)
    shuffle = torch.randperm(data.size()[0])
    data = data[shuffle]
    target = target[shuffle].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(data_source.size(0) - 1):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    # record every steps time
    # record time for every step
    start_time = datetime.now()
    inbatch_step = torch.randint(1,args.ngrams,(1,1)).item()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(train_data.size(0) - 1)):
        data, targets = get_batch(train_data, i,steps=inbatch_step)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = (datetime.now() - start_time).microseconds/1000
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.10f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, train_data.size(0), lr,
                elapsed/ args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = datetime.now()
    print("Total Encoder Time: {}s, Hidden Layer Time: {}s, Decoder Time: {}s, Softmax Time:{}s"
          .format(model.time_encoder/1e6,model.time_hidden/1e6,model.time_decoder/1e6,model.time_softmax/1e6))


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = datetime.now()
        print('-' * 89)
        print('Epoch {} start at {}'.format(epoch, epoch_start_time.strftime("%Y-%m-%d %H:%M:%S")))
        print('-' * 89)
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (datetime.now() - epoch_start_time).seconds,
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        elif best_val_loss <= 1.05*val_loss:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    print('Training Epoch {} end at {}'.format(epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    # if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
    #     model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
print('-' * 89)
print('Training end at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
print('-' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
