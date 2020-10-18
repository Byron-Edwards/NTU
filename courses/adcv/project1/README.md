## AI6126 Project 1: CelebA Facial Attribute Recognition Challenge
BAI WEN  
G1903363K
wbai001@e.ntu.edu.sg

### Private test result
The prediction of the private test dataset `predictions.txt` is in the `./checkpoints/` folder. It contains 13233 lines. Each line contains the image file name and the 40 predicted attributes.

### Third party library used
Base Code: https://github.com/d-li14/face-attribute-prediction
Data Augmentation: https://github.com/sthalles/SimCLR


### Folder structure

```python
├── ADCV.pdf  # short report
├── celeba.py  # dataset class used to load data
├── checkpoints  # checkpoints including log and predictions
│   ├── log.eps
│   ├── log.txt
│   └── predictions.txt   # prediction file for the private test dataset
├── data_aug  # data augmentation scripts
├── file_partition.sh   # script used to split the list_eval_partition.txt
├── main.py  # main script
├── models  # model scripts
├── README.md  # This readme 
└── utils  # utils scripts
```

### How to test
1. Download and unzip the private dataset into the root folder `./testset/`. If you want to use another test dataset, please keep the folder structure the same as the private test dataset we used.
2. Download my trained model and put it into `./checkpoints/` (please contact me if you need my trained model as it is too large to upload)
3. run the main script: `python main.py -pt --resume checkpoints/model_best.pth.tar`