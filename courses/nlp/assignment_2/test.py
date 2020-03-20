from data_prepare import *
import numpy as np
from torch.autograd import Variable


def test(model, word_to_id, char_to_id, tag_to_id):
    if not parameters['reload']:
        # reload the best model saved from training
        model.load_state_dict(torch.load(model_name))

    model_testing_sentences = [
        'President Donald Trump has said the US has "a tremendous testing set up where people coming in have to be tested',
        'It comes after the Jack Ma Foundation and the Alibaba Foundation last week announced that they had prepared 500,000 testing kits and 1 million masks to be sent to America.']

    # parameters
    lower = parameters['lower']

    # preprocessing
    final_test_data = []
    for sentence in model_testing_sentences:
        s = sentence.split()
        str_words = [w for w in s]
        words = [word_to_id[lower_case(w, lower) if lower_case(w, lower) in word_to_id else '<UNK>'] for w in str_words]

        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id] for w in str_words]

        final_test_data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
        })

    # prediction
    predictions = []
    print("Prediction:")
    print("word : tag")
    for data in final_test_data:
        words = data['str_words']
        chars2 = data['chars']

        d = {}

        # Padding the each word to max word size of that sentence
        chars2_length = [len(c) for c in chars2]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        for i, c in enumerate(chars2):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))

        # We are getting the predicted output from our model
        if use_gpu:
            val, predicted_id = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val, predicted_id = model(dwords, chars2_mask, chars2_length, d)

        pred_chunks = get_chunks(predicted_id, tag_to_id)
        temp_list_tags = ['NA'] * len(words)
        for p in pred_chunks:
            temp_list_tags[p[1]] = p[0]

        for word, tag in zip(words, temp_list_tags):
            print(word, ':', tag)
        print('\n')
