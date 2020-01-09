# About

This folder contains my experiments on automatically finding 
scaling parameters for different tight binding models: here graphene
and bilayer-graphene. There exists exact scaling formula for graphene
and bilayer graphene [1], however, this code can be used to find scaling 
for models which does not have continuum formulation e.g. graphene with 
Spin-Orbit interaction. 

# Requirements 
```
tensorflow >= '2.1.0-dev20200105'
```

# Things didn't work

* Newton second order optimizer was not stable, simple SGD with momentum was working well
* Problems with more than 3 parameters may not be stable, too 
    much local minimas to find the best answer. 

# References:

[1] Scalable Tight-Binding Model for Graphene https://arxiv.org/pdf/1407.5620.pdf  