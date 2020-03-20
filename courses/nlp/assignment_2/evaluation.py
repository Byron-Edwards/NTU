import torch
import numpy as np
from torch.autograd import Variable

from Parameters import *
from data_prepare import get_chunks


def evaluating(model, datas, tag_to_id, best_F, dataset="Train"):
    '''
    The function takes as input the model, data and calcuates F-1 Score
    It performs conditional updates
     1) Flag to save the model
     2) Best F-1 score
    ,if the F-1 score calculated improves on the previous F-1 score
    '''
    with torch.no_grad():
        # Initializations
        prediction = []  # A list that stores predicted tags
        save = False  # Flag that tells us if the model needs to be saved
        new_F = 0.0  # Variable to store the current F1-Score (may not be the best)
        correct_preds, total_correct, total_preds = 0., 0., 0.  # Count variables

        for data in datas:
            ground_truth_id = data['tags']
            words = data['str_words']
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
                val, out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
            else:
                val, out = model(dwords, chars2_mask, chars2_length, d)
            predicted_id = out

            # We use the get chunks function defined above to get the true chunks
            # and the predicted chunks from true labels and predicted labels respectively
            lab_chunks = set(get_chunks(ground_truth_id, tag_to_id))
            lab_pred_chunks = set(get_chunks(predicted_id, tag_to_id))

            # Updating the count variables
            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)

        # Calculating the F1-Score
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        new_F = 2 * p * r / (p + r) if correct_preds > 0 else 0

        print("{}: new_F: {} best_F: {} ".format(dataset, new_F, best_F))


        # If our current F1-Score is better than the previous best, we update the best
        # to current F1 and we set the flag to indicate that we need to checkpoint this model

        if new_F > best_F:
            best_F = new_F
            save = True

        return best_F, new_F, save
