import pandas as pd
import torch
import data
from scipy.stats import spearmanr
from scipy import spatial
import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model to calculate word pair similarity')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--wordpair', type=str, default='./data/combined.csv',
                    help='model checkpoint to use')
parser.add_argument('--out', type=str,default='./spearmanr.csv',
                    help='the output file path')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda" if args.cuda else "cpu")

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)


def idx2word(n):
    return corpus.dictionary.idx2word[n]


with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

raw_dt = pd.read_csv("./data/combined.csv")
dt = raw_dt.replace({"Word 1": corpus.dictionary.word2idx, "Word 2": corpus.dictionary.word2idx})
dt = dt.loc[(dt["Word 1"].apply(type) == int) & (dt["Word 2"].apply(type) == int), ["Word 1", "Word 2"]].astype(int)
with torch.no_grad():
    out = torch.tensor(dt.values,dtype = torch.long)
    out = model.encoder(out.cuda()).cpu().numpy()
    corr = []
    cos = []
    for i in out:
        corr.append(spearmanr(i[0],i[1]).correlation)
        cos.append(1 - spatial.distance.cosine(i[0],i[1]))
    dt["Spearmanr"] = corr
    dt["Cosin"] = cos
out = raw_dt.merge(dt["Spearmanr"],how='outer',right_index=True,left_index=True)
out = out.merge(dt["Cosin"],how='outer',right_index=True,left_index=True)
print(out[["Human (mean)","Spearmanr","Cosin"]].corr(method= "spearman"))
out.to_csv(args.out,index=False)


