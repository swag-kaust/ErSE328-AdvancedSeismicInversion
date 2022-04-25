# Assignment 5

In this assignment we will impliment a similtanious linear/non-linear inversion. 
 

The objective function for FWI is non-linear and is written as:

<img src="https://latex.codecogs.com/svg.image?J(m)&space;=&space;\left\|L(m)&space;-&space;d^{obs}&space;\right\|" title="https://latex.codecogs.com/svg.image?J(m) = \left\|L(m) - d^{obs} \right\|" />

where F indicates the wave-equation operator. Here, the relation between F and the model (m) is non-linear.

Using the Born modelling, instead of the wave-equation, we can update the high-wavenumber components of the model (aka model perturbation) using the objective: 

<img src="https://latex.codecogs.com/svg.image?J(\delta&space;m)\left\|L(m)&space;-&space;d^{obs}&space;\right\|" title="https://latex.codecogs.com/svg.image?J(\delta m)\left\|L(m) - d^{obs} \right\|" />

Where L represents the Born modelling operator. The relationship between the Born modelling and the model perturbation (<img src="https://latex.codecogs.com/svg.image?\delta&space;m" title="https://latex.codecogs.com/svg.image?\delta m" />) is linear, hence the linearized inversion.


The two inversion can be combined and implimented similtaniously by estimating the model perturbation update at each nonlinear iteration:

<img src="https://latex.codecogs.com/svg.image?m&space;=&space;m_o&space;&plus;&space;\delta&space;m" title="https://latex.codecogs.com/svg.image?m = m_o + \delta m" />






### Tasks: 
1. Use the notebook to impliment the simitanous linear/non-linear inversion.
2. Compare with applying only the non-linear inversion (FWI) in terms of accuracy, cost/speed. 
