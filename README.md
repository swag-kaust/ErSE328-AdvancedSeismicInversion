![erse328asi](logo.png)

This repo contains materials for the course ErSE328 Advanced Seismic Inversion course taught by Professor Tariq Alkhalifah in King Abdullah University of Science. An inversion example is shown below:

# Getting started

Throughout the computational part of the course, we will mainly be utilizing the Deepwave Python library which you can access from their [repository](https://git@github.com:alaliaa/ErSE328-AdvancedSeismicInversion.git). To get yourself started, you can directly learn the Deepwave fundamentals from their [documentations](https://ausargeo.com/deepwave/).

To install the environment, run the following command:
```
sh install_env.sh
```
It will take some time, but if, in the end, you see the word `Done!` on your terminal, you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate erse328asi
```

# Run a notebook on KAUST's Ibex

First connect to Ibex using your KAUST credential.

```
ssh USERNAME@glogin.ibex.kaust.edu.sa
```
Then, clone this repository and install the `erse328asi` conda environment. If it is installed please go ahead with the following command. If not, please refer to this [documentation](https://docs.anaconda.com/free/miniconda/) to install conda. Inside the `ErSE328-AdvancedSeismicInversion` folder, run the following command to submit the slurm jupyter notebook request.

```
sbatch erse328asi_notebook.slurm
```

You can connect from your workstation to access the notebook with the instructions from the output slurm job. The file is in the format of `slurm-JOBID.out`.

```
ssh -L 6789:GPUID:6789 USERNAME@glogin.ibex.kaust.edu.sa
```

where the `JOBID` and `GPUID` are the unique identifiers from the slurm output job request. Then, access the `http://localhost:6789/` link from your workstation and fill in the credential from the `slurm-JOBID.out`.

# Assignments 
Assignment #  | Due date     | Objectives
------------- | -------------| ------------
[Assignment 0](./00_introduction) | 04/03/2024 | Ensure the Deepwave package is installed properly.
[Assignment 1](./01_marmousi) | 07/03/2024 | Perform FWI to the Marmousi and suggest ways to improve the results.
[Assignment 2](./02_multiscale) | 24/03/2024 | Perform multiscale FWI to improve the results.
[Assignment 3](./03_regularization) | 05/05/2024 | Remedy the ill-posedness of FWI using regularization.
[Assignment 4](./04_multiparameter) | 05/05/2024 | Moving beyond acoustic media assumption.
