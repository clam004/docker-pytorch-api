# docker-pytorch-api
Deploying PyTorch as a RESTAPI using Docker and FastAPI with CUDA support

## Setup

- Ubuntu 18.04.3 LTS (bionic)
- Python 3.8
- Cuda 10.1
- 

### Running just the model API without Docker

lets start from the very beginning, before any understanding of Docker, just good old python virtual environments, these terminal commands assume you have python 3.8 installed as python3.8 and get you to the old fashioned way of setting up a virtual python environment with the machine learning libraries we need such torch and sklearn

```console
$ pip3 install virtualenv

$ virtualenv venv --python=python3.8

$ source venv/bin/activate

(venv) $ pip3 install -r requirementsDS.txt

(venv) $ jupyter notebook
```

navigate to notebook/Model.ipynb to train a toy model
and save the model and the preprocessing module for later use by the API

```python
from joblib import dump

torch.save(model.state_dict(), '../model/model.pt')

dump(scaler, '../model/scaler.joblib', compress=True)
```

in the app/ directory

```console
(venv) $  python main.py
```

Navigate to http://localhost:8080/docs to test the API

If you want to know which files are needed for the model API, it is only those files
used by main.py or the scripts imported to main.py

### Automated testing

in the app/ directory

```console
(venv) $  python -m pytest
```

### Deploying with Docker

Here we build the model API into a docker image. All the dependencies our model API needs will be contained inside the image and will have no conflict with other APIs or applications when we scale it. 

What CUDNN am I using?

```
(venv) $ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

```

the below is am example where the cudnn verison is CUDNN 7.6.3
<img src="https://www.programmerall.com/images/696/64/6449e5a8805928f086215fde26801800.png" height=200, width=500>

What CUDA version am I using ?

```
nvcc --version
```
the below is CUDA 10.1
<img src="https://i0.wp.com/varhowto.com/wp-content/uploads/2020/07/Use-nvcc-version-to-check-cuda-version.png?w=606&ssl=1" height=200, width=500>


Thank you to [Ming](https://github.com/ming0070913/example-ml-project) for the original version of this tutorial

Thank you to [Nikolai Janakiev](https://janakiev.com/blog/pytorch-iris/) 
