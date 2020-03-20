#bin/bash
python cnn_word_encoder.py &
python cnn_word_encoder.py --multi_cnn 2 &
python cnn_word_encoder.py --multi_cnn 3 &
python cnn_word_encoder.py --dilation 1  &
#python cnn_word_encoder.py --char_mode LSTM &
#python cnn_word_encoder.py --crf 0  &
