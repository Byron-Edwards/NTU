import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from Parameters import *


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 char_to_ix=None, pre_word_embeds=None, char_out_dimension=25, char_embedding_dim=25, char_lstm_dim=25,
                 use_gpu=False, use_crf=True, char_mode='CNN',multi_cnn=1, dilation=False):
        '''
        Input parameters:

                vocab_size= Size of vocabulary (int)
                tag_to_ix = Dictionary that maps NER tags to indices
                embedding_dim = Dimension of word embeddings (int)
                hidden_dim = The hidden dimension of the LSTM layer (int)
                char_to_ix = Dictionary that maps characters to indices
                pre_word_embeds = Numpy array which provides mapping from word embeddings to word indices
                char_out_dimension = Output dimension from the CNN encoder for character
                char_embedding_dim = Dimension of the character embeddings
                use_gpu = defines availability of GPU,
                    when True: CUDA function calls are made
                    else: Normal CPU function calls are made
                use_crf = parameter which decides if you want to use the CRF layer for output decoding
        '''

        super(BiLSTM_CRF, self).__init__()

        # parameter initialization for the model
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension
        self.char_mode = char_mode
        self.char_lstm_dim= char_lstm_dim
        self.multi_cnn = multi_cnn
        self.dilation = dilation

        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim

            # Initializing the character embedding layer
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)

            # Performing LSTM encoding on the character embeddings
            if self.char_mode == 'LSTM':
                self.char_lstm = nn.LSTM(char_embedding_dim, self.char_lstm_dim, num_layers=1, bidirectional=True)
                init_lstm(self.char_lstm)

            # Performing CNN encoding on the character embeddings
            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                           kernel_size=(3, char_embedding_dim), padding=(2, 0))

        # Creating Embedding layer with dimension of ( number of words * dimension of each word)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            # Initializes the word embeddings with pretrained word embeddings
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        # Initializing the dropout layer, with dropout specificed in parameters
        self.dropout = nn.Dropout(parameters['dropout'])


        # Lstm Layer:
        # input dimension: word embedding dimension + character level representation
        # bidirectional=True, specifies that we are using the bidirectional LSTM
        if self.char_mode == 'LSTM':
            kernal_height = embedding_dim + self.char_lstm_dim * 2
        elif self.char_mode == "CNN":
            kernal_height = embedding_dim + self.out_channels

        # self.lstm = nn.LSTM(embedding_dim + self.char_lstm_dim * 2, hidden_dim, bidirectional=True)
        # self.lstm = nn.LSTM(embedding_dim + self.out_channels, hidden_dim, bidirectional=True)
        if self.multi_cnn == 2:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=hidden_dim,
                          kernel_size=(3, kernal_height), padding=(1, 0)),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2,
                          kernel_size=(5, 1), padding=(2, 0)),
                nn.ReLU(),
            )
        elif self.multi_cnn == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=hidden_dim,
                          kernel_size=(3, kernal_height), padding=(1, 0)),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                          kernel_size=(5, 1), padding=(2, 0)),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2,
                          kernel_size=(7, 1), padding=(3, 0)),
                nn.ReLU(),
            )
        elif self.dilation:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=hidden_dim, dilation=(1, 1),
                          kernel_size=(3, kernal_height), padding=(1, 0)),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, dilation=(2, 1),
                          kernel_size=(3, 1), padding=(2, 0)),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, dilation=(3, 1),
                          kernel_size=(3, 1), padding=(3, 0)),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=hidden_dim * 2,
                          kernel_size=(3, kernal_height), padding=(1, 0)),
                nn.ReLU(),
            )


        # Initializing the lstm layer using predefined function for initialization
        # init_lstm(self.lstm)

        # Linear layer which maps the output of the bidirectional LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)

        # Initializing the linear layer using predefined function for initialization
        init_linear(self.hidden2tag)

        if self.use_crf:
            # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
            # Matrix has a dimension of (total number of tags * total number of tags)
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))

            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    # assigning the functions, which we have defined earlier
    # _score_sentence = score_sentences
    # _get_lstm_features = get_lstm_features
    # _forward_alg = forward_alg
    # viterbi_decode = viterbi_algo
    # neg_log_likelihood = get_neg_log_likelihood
    # forward = forward_calc

    def get_lstm_features(self, sentence, chars2, chars2_length, d):
        if self.char_mode == 'LSTM':

            chars_embeds = self.char_embeds(chars2).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)

            # chars_embeds = [x for _, x in sorted(zip(chars2_length, chars_embeds), reverse=True)]
            # packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, sorted(chars2_length, reverse=True))
            # packed = [x for _, x in sorted(zip(np.argsort(chars2_length)[::-1], packed))]

            lstm_out, _ = self.char_lstm(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)

            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))

            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.cuda()

            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat(
                    (outputs[i, index - 1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))

            chars_embeds = chars_embeds_temp.clone()

            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]
        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)

            ## Creating Character level representation using Convolutional Neural Netowrk
            ## followed by a Maxpooling Layer
            chars_cnn_out3 = self.char_cnn3(chars_embeds)

            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,kernel_size=(chars_cnn_out3.size(2), 1))\
                .view(chars_cnn_out3.size(0), self.out_channels)

            ## Loading word embeddings
        embeds = self.word_embeds(sentence)

        ## We concatenate the word embeddings and the character level representation
        ## to create unified representation for each word

        embeds = torch.cat((embeds, chars_embeds), 1)
        # embeds = embeds.unsqueeze(1)
        embeds = embeds.unsqueeze(0).unsqueeze(0)

        ## Dropout on the unified embeddings
        embeds = self.dropout(embeds)
        # print("embedding after cat :{}".format(embeds.shape))

        ## Word lstm
        ## Takes words as input and generates a output at each step
        # lstm_out, _ = self.lstm(embeds)
        lstm_out = self.cnn(embeds)
        # print("lstm_out :{}".format(lstm_out.shape))

        # lstm_out = nn.functional.max_pool2d(lstm_out,kernel_size=(lstm_out.size(2), 1))
        # print("lstm_out :{}".format(lstm_out.shape))


        ## Reshaping the outputs from the lstm layer
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
        lstm_out = lstm_out.permute(2, 0,1,3).view(len(sentence)*lstm_out.size(0),self.hidden_dim*2)
        # print("lstm_out :{}".format(lstm_out.shape))

        ## Dropout on the lstm output
        lstm_out = self.dropout(lstm_out)

        ## Linear layer converts the ouput vectors to tag space
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def neg_log_likelihood(self, sentence, tags, chars2, chars2_length, d):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats = self.get_lstm_features(sentence, chars2, chars2_length, d)

        if self.use_crf:
            forward_score = self.forward_alg(feats)
            gold_score = self.score_sentences(feats, tags)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores

    def score_sentences(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def forward_alg(self, feats):
        '''
        This function performs the forward algorithm explained above
        '''
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.

        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()

        # Iterate through the sentence
        for feat in feats:
            # broadcast the emission score: it is the same regardless of
            # the previous tag
            emit_score = feat.view(-1, 1)

            # the ith entry of trans_score is the score of transitioning to
            # next_tag from i
            tag_var = forward_var + self.transitions + emit_score

            # The ith entry of next_tag_var is the value for the
            # edge (i -> next_tag) before we do log-sum-exp
            max_tag_var, _ = torch.max(tag_var, dim=1)

            # The forward variable for this tag is log-sum-exp of all the
            # scores.
            tag_var = tag_var - max_tag_var.view(-1, 1)

            # Compute log sum exp in a numerically stable way for the forward algorithm
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)  # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def viterbi_algo(self, feats):
        '''
        In this function, we implement the viterbi algorithm explained above.
        A Dynamic programming based approach to find the best tag sequence
        '''
        backpointers = []
        # analogous to forward

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()  # holds the backpointers for this step
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]  # holds the viterbi variables for this step
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            if self.use_gpu:
                viterbivars_t = viterbivars_t.cuda()

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence, chars, chars2_length, d):
        '''
        The function calls viterbi decode and generates the
        most probable sequence of tags for the sentence
        '''

        # Get the emission scores from the BiLSTM
        feats = self.get_lstm_features(sentence, chars, chars2_length, d)
        # viterbi to get tag_seq

        # Find the best path, given the features.
        if self.use_crf:
            score, tag_seq = self.viterbi_algo(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().numpy())

        return score, tag_seq


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    """
    Initialize lstm

    PyTorch weights parameters:

        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size * hidden_size)`

        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size * hidden_size)`
    """

    # Weights init for forward layer
    for ind in range(0, input_lstm.num_layers):
        ## Gets the weights Tensor from our model, for the input-hidden weights in our current layer
        weight = eval('input_lstm.weight_ih_l' + str(ind))

        # Initialize the sampling range
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))

        # Randomly sample from our samping range using uniform distribution and apply it to our current layer
        nn.init.uniform(weight, -sampling_range, sampling_range)

        # Similar to above but for the hidden-hidden weights of the current layer
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -sampling_range, sampling_range)

    # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -sampling_range, sampling_range)

    # Bias initialization steps

    # We initialize them to zero except for the forget gate bias, which is initialized to 1
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))

            # Initializing to zero
            bias.data.zero_()

            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            # Similar for the hidden-hidden layer
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        # Similar to above, we do for backward layer if we are using a bi-directional LSTM
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def log_sum_exp(vec):
    '''
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * tagset_size
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    '''
    This function returns the max index in a vector
    '''
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def to_scalar(var):
    '''
    Function to convert pytorch tensor to a scalar
    '''
    return var.view(-1).data.tolist()[0]


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
