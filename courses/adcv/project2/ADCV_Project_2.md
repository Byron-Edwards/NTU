## AI6126 Project 2: DIV2K Single Image Super-Resolution Challenge
BAI WEN  
G1903363K
wbai001@e.ntu.edu.sg

### Private test result
The Supre Resolution images of the private test dataset is in the `./private_result/` folder. It contains 80 images with the same file name in the private dataset.

The model checkpoint is located in `./Codes/experiments/001_MSRResNet_x4_f64b20_DIV2K_1000k_B16G1_wandb/models/net_g_latest.pth`

### Third party library used
Codebase: [BasicSR](https://github.com/xinntao/BasicSR)


### Folder structure
```python
├── ADCV_Project_2.pdf  # short report
├── ADCV_Project_2.md  # This readme 
├── private_result  # Images from Private dataset after Super Resolution 
└── Codes # The code
     ├── experiments/001_MSRResNet_x4_f64b20_DIV2K_1000k_B16G1_wandb/models/net_g_latest.pth  # model checkpoint
     ├── datasets/Mini-DIV2K  # Training and validation dataset
     └── datasets/LR # private dataset
```
For details of the Github of [BasicSR](https://github.com/xinntao/BasicSR)

### How to test
1. Download and unzip the private dataset into the folder `./Codes/datasets`. If you want to use another test dataset, please keep the folder structure the same as the private test dataset I used.
2. cd into the `Codes` folder and run the test command:
```bash
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4_woGT.yml
```