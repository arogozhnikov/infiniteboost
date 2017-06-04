# InfiniteBoost

Reseach code for a paper:  <br />
InfiniteBoost: building infinite ensembles with gradient descent.
[TODO LINK to a paper](TODO link to a paper)

## Description

InfiniteBoost is an approach to building ensembles which combines best sides of random forest and gradient boosting. 

Trees in the ensemble encounter mistakes done by previous trees, but due to modified scheme of encountering contributions
the ensemble converges to the limit, thus avoiding overfitting, just as random forest.

## Reproducing research

Research is performed in jupyter notebooks 
(if you're not familiar, read [why Jupyter notebooks are awesome](http://arogozhnikov.github.io/2016/09/10/jupyter-features.html)).

You can use docker image `arogozhnikov/pmle:0.01` from docker hub. 
Dockerfile is stored in this repository (ubuntu 16 + basic sklearn stuff).

To run the environment (sudo is needed on linux):
```bash
sudo docker run -it --rm -v /YourMountedDirectory:/notebooks -p 8890:8890 arogozhnikov/pmle:0.01
```
(and open `localhost:8890` in the browser).


## InfiniteBoost package

Self-written minimalistic implementation of trees as used for experiments against boosting.

Specific implementation was used to compare with random forest and based on the trees from scikit-learn package. 

Code written in python 2, some critical functions in fortran, so you need `gfortran + openmp` installed 
before installing the package (or simply use docker image).

```bash
pip install numpy
pip install .
nosetests tests
```
