# MCNCC

Shoeprint matching algorithm using multiple feature channels from a pretrained neural network (googlenet) and normalized cross correlation

Steps:

- make sure you have access to a gpu and have cuda and the cuda toolkit installed.
- download the FID-300 dataset and create a new project folder containing the python file and the "datasets" folder

<img src="Folder_structure.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
     
- (optionally) in the datasets folder create two additonal folders with subsets
- install all the necessary packages: matplotlib, numpy, torch,..
- after running the program, a .npy file is created storing the correlation matrix
- use the cmc function in order to create cmc-plots from your score-files


