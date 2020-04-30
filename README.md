# MCNCC

Shoeprint matching algorithm using multiple feature channels from a pretrained neural network (googlenet) and normalized cross correlation

Steps:

- make sure you have access to a gpu and have cuda and the cuda toolkit installed.
- download the FID-300 dataset and create a new project folder containing the python file and the "datasets" folder

<img src="Folder_structure.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
     
- (optionally) in the datasets folder create two additonal folders with subsets, if you don't want to run through the whole dataset (depending on your gpu, this can take a long time). For example the first 50 track images and the corresponding reference images (The right labels are in the FID-300 dataset).
- depending on your folderstructure, change the lines 22, 24, 25 in mcncc.py accordingly
- create a virtual environment for this project and install all the necessary packages: matplotlib, numpy, torch,..
- after running the program, a .npy file is created storing the correlation matrix (rows: number of tracks in the chosen track folder, columns: number of reference images in the chosen reference image folder)

- use the cmc function in order to create cmc-plots from your correlation score-files


