# Assignment 4  

In this assignment we will impliment a regularized FWI. 
For an ill-posed optimization problem such as FWI, regularization often help to impose some known feature to the inversion, such as smoothness, blockness ... etc.   

The objective function for a regularized inversion has the form, 

<img src="https://latex.codecogs.com/png.image?\dpi{110}J(m)&space;=\left\|d^{obs}&space;-&space;F(m)&space;\right\|&space;&plus;&space;\alpha&space;&space;\mathbf{R}" title="https://latex.codecogs.com/png.image?\dpi{110}J(m) =\left\|d^{obs} - F(m) \right\| + \alpha \mathbf{R}" />, 

where <img src="https://latex.codecogs.com/png.image?\dpi{110}\alpha" title="https://latex.codecogs.com/png.image?\dpi{110}\alpha" />  is a coefficient controling the tradeoff between the objectives and **R** represents the regularization.  

In this assignment we will impliment two regularizations that are Tikhonov and total variation (TV). The formula for Tikhonov is: 

<img src="https://latex.codecogs.com/png.image?\dpi{110}R_{TV}&space;=&space;\left\|&space;L&space;\mathbf{m}&space;\right\|^2_2&space;" title="https://latex.codecogs.com/png.image?\dpi{110}R_{TV} = \left\| L \mathbf{m} \right\|^2_2 " />, 

where L is designed to penalize the function. In our assignment we will chose L to be the first order derivative. 

The TV regularization reads: 

<img src="https://latex.codecogs.com/png.image?\dpi{110}R_{TV}&space;=&space;\left\|&space;&space;\nabla&space;\mathbf{m}&space;\right\|_1&space;" title="https://latex.codecogs.com/png.image?\dpi{110}R_{TV} = \left\| \nabla \mathbf{m} \right\|_1 " />




### Tasks: 
1. Use the notebook (a) to apply conventional FWI on a layered model with an anomaly. Then, apply the inversion with Tikhonov regulurization and total variation. Compare the results and comments on the differences between the two regulrizations. What is the role of <img src="https://latex.codecogs.com/png.image?\dpi{110}\alpha" title="https://latex.codecogs.com/png.image?\dpi{110}\alpha" /> and how you chose it ? 
2. Look into letruture for some novel objective functions that are immune to cycle-skipping, impliment one of them in the second notebook and compare it with the conventional L2 objective. (bonus: shows the convexity of the objective function and compare it with L2 )

     
