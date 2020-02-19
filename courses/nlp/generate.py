###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='500',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

# is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
# if not is_transformer_model:
#     hidden = model.init_hidden(1)
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w',encoding='utf-8') as outf:
    with torch.no_grad():  # no tracking history
        print('-' * 89)
        input_vector = input('Please input your words: ')
        if len(input_vector) == 0:
            input_vector = "You will never know what happened here"
        outf.write('-' * 45 + 'Your input' + '-' * 45 + "\n")
        outf.write(input_vector + " \n")
        outf.write('-' * 45 + 'Your input' + '-' * 45 + "\n")
        input_vector = torch.tensor([corpus.dictionary.word2idx[i] for i in input_vector.split()], dtype=torch.long).unsqueeze(dim=0).to(device)[:,-model.ngram:]
        output_word = ""
        for i in range(args.words):
            # if is_transformer_model:
            output = model(input_vector)
            word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input_vector = torch.cat([input_vector, word_tensor], 1)[:,-model.ngram:]

            word = corpus.dictionary.idx2word[word_idx]
            output_word = output_word + " " + word
            outf.write(word + ('\n' if i % 20 == 19 else ' '))
            

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
        print('The following are generated words: \n {}'.format(output_word))
