# docker-pytorch-api
Deploying PyTorch as a RESTAPI using Docker and FastAPI with CUDA support

## Setup

- Ubuntu 18.04.3 LTS (bionic)
- Python 3.8

lets start from the very beginning, before any understanding of Docker, just good old python virtual environments, these terminal commands assume you have python 3.8 installed as python3.8 and get you to the old fashioned way of setting up a virtual python environment with the machine learning libraries we need such torch and sklearn

```console
pip3 install virtualenv

virtualenv venv --python=python3.8

source venv/bin/activate

pip3 install -r requirementsDS.txt

jupyter notebook
```

navigate to notebook/Model.ipynb to train a toy model
and save the preprocessing module for later use by the API

## Credit and References

Thank you to [Ming](https://github.com/ming0070913/example-ml-project) for the original version of this tutorial

Thank you to [Nikolai Janakiev](https://janakiev.com/blog/pytorch-iris/) 
