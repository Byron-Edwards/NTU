# AI6126 Homework2
Name: *BAIWEN* 
Matric number: *G1903363K*

### Question 1: 
Explain why is scaling and shifting often applied after batch normalization?
#### Answer1:
Normalizing the mean and standard deviation of a unit can reduce the expressive power of the neural network containing that unit. Replacing the batch of hidden unit activations $H$ with $\gamma H+\beta$ can maintain the expressive power of the network. The newly introduced  parametrization can represent the same family of functions of the input as the old parametrization. Besides, in the new parameters, the mean of $\gamma H^{\prime} + \beta$ determined solely by $\beta$ and will not impacted by the complicated interaction between the parameters in the layers below $H$

### Question 2:
Describe the main changes, e.g., architecture change or new losses, made in Faster R-CNN in comparison to Fast R-CNN


### Question 3:
Describe the differences between the operation of RoIPool and RoIAlign. Explain why RoIAlign is preferred over RoIPool

### Question 4:
1. Explain why the encoder-decoder architecture is widely used in semantic segmentation tasks. 
2. Does the plain encoder-decoder architecture have potential drawbacks? If so, how can we fix them?

### Question 5:
When we apply consecutive 1-dilated, 2-dilated, 4-dilated and 8-dilated
3x3 convolution, what is the final receptive field?

### Question 6:
1. Even though dilated convolution improves upon standard convolution, what are the potential hard cases for dilated convolution? 
2. How will you further improve upon dilated convolution? 
