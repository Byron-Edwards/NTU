# Answer Sheet

## Repo Structure

- [cnn_word_encoder.py](cnn_word_encoder.py) contains the data processing and model training.
- [Model.py](Model.py) contains all the models required by the assignment
- [data_prepare.py](data_prepare.py) contains the functions of data preprocessing
- [Parameters.py](Parameters.py) contains all the parameters and hyperparameters of the training
- [evaluation.py](evaluation.py) contains the functions used to evaluate the model performance on different datasets
- [batch_run.sh](batch_run.sh) is bash script used to execute different training setting parallely.
- [models](models) contains the trained model and training logs, all the observations for the assignments are retrieved from the training logs.(For the model file, use this link to download if necessary)

## Conclusions

With the same hyper parameter, I have the following observation based on the experiments results:

- when using single layer CNN encoder, the performance of the network with CNN char encoder is similar to that with LSTM char encoder. While, if no use the CRF, the performance drops lot (0.82 -> 0.77)
- Performance become better with more layers
- Dilation CNN can achieve similar performance with less model parameters

## Performance Summary

- Question (iv): Single layer CNN word-level encoder. [Logfile](./models/_20200319_151650_CNN_1_0_1.log)

```python
Training Start at 2020/03/19 15:17:11
Number of Model Parameters: 1911455
Dev best_F: 0.8874183568236508
Test best_F: 0.8259550160656909
Training End at 2020/03/19 18:57:57
Training Time 13245 s
```

Question (v): Single layer CNN word-level encode with LSTM char encoder [Logfile](./models/_20200319_191939_LSTM_1_0_1.log)

```python
Training Start at 2020/03/19 19:19:55
Number of Model Parameters: 1949955
Dev best_F: 0.895930677087758
Test F with best Dev: 0.8243786549707602
Training End at 2020/03/19 22:58:37
Training Time 13121 s
```

Question (vi): 2 Layer CNN word-level encoder. [Logfile](./models/_20200320_000019_CNN_2_0_1.log)

```python
Training Start at 2020/03/20 00:00:36
Number of Model Parameters: 2236655
Dev best_F: 0.9253503184713375
Test with best Dev best_F: 0.8777767784872741
Training End at 2020/03/20 01:58:37
Training Time 7081 s
```

Question (vi): 3 Layer CNN word-level encoder. [Logfile](./models/_20200319_151650_CNN_3_0_1.log)

```python
Training Start at 2020/03/19 15:17:11
Number of Model Parameters: 2596855
Dev best_F: 0.9185689948892675
Test F with Best Dev: 0.8729987407807159
Training End at 2020/03/19 19:09:23
Training Time 13931 s
```

Question (vii): 3 Layer CNN word-level encoder with dilaton. [Logfile](./models/_20200319_151650_CNN_1_1_1.log)

```python
Training Start at 2020/03/19 15:17:11
Number of Model Parameters: 2196855
Dev best_F: 0.9166101694915254
Test F with best Dev: 0.8602842585143469
Training End at 2020/03/19 19:10:01
Training Time 13970 s
```

Question (viii): Single Layer CNN word-level encoder without CRF. [Logfile](./models/_20200319_200142_CNN_1_0_0.log)

```python
Training Start at 2020/03/19 20:01:59
Number of Model Parameters: 1911094
Dev best_F: 0.8406180285879534
Test F with best Dev: 0.7751443592174437
Training End at 2020/03/19 20:48:46
Training Time 2806 s
```

## Models

The following are the cnn model used for assignment, For the details please refer to [Model.py](Model.py)

```python
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
```
