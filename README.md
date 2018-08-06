# MoDL
MoDL: Model-Based Deep Learning Architecture for Inverse Problems

### Reference paper: 
Link: https://arxiv.org/abs/1712.02862

#### Dependencies

We have tested the code in Anaconda python 2.7 and 3.6. The code should work with Tensorflow-1.7 onwards.

#### Dataset
We have released the parallel imaging dataset used in this paper. You can download the dataset from the below link:

 **Download Link** :  https://drive.google.com/file/d/1Z-P5UKGbfD0caq_AMX6UdfXYmPsFHdmD/view?usp=sharing

This dataset consist of parallel magnetic resonance imaging (MRI) brain data of five human subjects. Four of which are used during training of the model and fifth subject is used during testing.
Above link contain fully sampled preprocessed data in numpy format for both training and testing. We also provide coil-sensitity-maps (CSM) pre-computed using E-SPIRIT algorithm. Total file size is 3 GB and contains following arrays:

`trnOrg`: This is complex arrary of 256x232x360 containing 90 slices from each of the 4 training subjects. 
        Each slice is of  spatial dimension 256x232. This is the original fully sampled data.
        
`trnCSM`: This is a complex array of 256x256x12x360 representing coil sensitity maps (csm). Here 12 represent number of coils.

`trnMask`: This is the random undersampling mask to do 6-fold acceleration. We use different mask for different slices.

`tstOrg`,`tstCSM`, `tstMask`: These are similar arrays for testing purpose.


#### How to run the code

First, ensure that Tensorflow 1.7 or higher version is installed and working with GPU. 
Second, you will need the `dataset` to run the code. You can download the dataset from the link provided above.
Third, just clone or download this reporsitory. The `tstDemo.py` file should run without any changes in the code.
Please ensure to keep the dataset in the same directory as `tstDemo.py` file or set the dataset path appropriately in the code.

#### Files description
The folder `TFmodels` contain the learned tensorflow model parameters. `tstDemo.py` will use it to read the model and run on the `dataset` downloadable from the above link.

`supportingFunctions.py`: This file contain some supporting functions to calculate the time, PSNR, and read the dataset.

#### Training Code
To be uploaded soon

#### Testing Code
`tstDemo.py` : This file contains the comments

#### Contact
The code is provided to support reproducible research. It may not be robust enough to work directly on your particular configuration of python. If the code is not working or some files are missing then
you may open an issue or directly email me at jnu.hemant@gmail.com


