# docker-pytorch-api
Deploying PyTorch as a RESTAPI using Docker and FastAPI with CUDA support

## Setup

- Ubuntu 18.04.3 LTS (bionic)
- Python 3.8
- Cuda 10.1
- cudnn7.6.4
- PyTorch 1.10.0

### Running just the model API without Docker

lets start from the very beginning, before any understanding of Docker, just good old python virtual environments, these terminal commands assume you have python 3.8 installed as python3.8 and get you to the old fashioned way of setting up a virtual python environment with the machine learning libraries we need such torch and sklearn

```console
you@you:/path/to/folder$ pip3 install virtualenv

you@you:/path/to/folder$ virtualenv venv --python=python3.8

you@you:/path/to/folder$ source venv/bin/activate

(venv) you@you:/path/to/folder$ pip3 install -r requirementsDS.txt

(venv) you@you:/path/to/folder$ jupyter notebook
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
(venv) you@you:/path/to/folder$  python main.py
```

Navigate to http://localhost:8080/docs to test the API

If you want to know which files are needed for the model API, it is only those files
used by main.py or the scripts imported to main.py

### Automated testing

in the app/ directory

```console
(venv) you@you:/path/to/folder$  python -m pytest
```

### Deploying with Docker

[Installing Docker on Ubuntu and some basic Docker commands](https://www.simplilearn.com/tutorials/docker-tutorial/how-to-install-docker-on-ubuntu)

Here we build the model API into a docker image. All the dependencies our model API needs will be contained inside the image and will have no conflict with other APIs or applications when we scale it. 

#### what are the specs for the Image we want to grab from DockerHub?

What CUDNN am I using?

```
(venv) you@you:/path/to/folder$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

```

the below is am example where the cudnn verison is CUDNN 7.6.3

<img src="https://www.programmerall.com/images/696/64/6449e5a8805928f086215fde26801800.png" height=200, width=500>

What CUDA version am I using ?

```
(venv) you@you:/path/to/folder$ nvcc --version
```
the below is CUDA 10.1

What Ubuntu version am I using?

```
(venv) you@you:/path/to/folder$ lsb_release -a
```
The below is version 18.04
```
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 18.04 LTS
Release:    18.04
Codename:   bionic
```

Tags for Nvidia GPU Docker Images

[NVIDIA Docker Image s](https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md)

<img src="https://i0.wp.com/varhowto.com/wp-content/uploads/2020/07/Use-nvcc-version-to-check-cuda-version.png?w=606&ssl=1" height=200, width=500>

If you have my exact specs you would choose

```
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
```
in your Dockerfile

#### Whats Dockerfile?

The Dockerfile turns this application into a container. Inside you will see a commented file
showing how first an Nvidia CUDA image is first built. Then apt-get and miniconda are used to
build python3.8, then PyTorch, then the API and model itself are loaded. Last,  set the entrypoint as /start.sh and expose port 80 of the image.

```
(venv) you@you:/path/to/docker-pytorch-api$ bash docker_build.sh
```
this may take awhile

you can also do 

```
docker build --compress -t ml/project1:latest .
```

but if you get an error like this

```
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock
```

try

```
sudo usermod -a -G docker $USER
```

```
(venv) you@you:/path/to/docker-pytorch-api$ bash docker_run_local.sh
```
and to see it running

```
(venv) you@you:/path/to/docker-pytorch-api$ docker ps
```

to stop it

```
(venv) you@you:/path/to/docker-pytorch-api$ docker stop <CONTAINER ID>
```



## Credit and references

Thank you to [Ming](https://github.com/ming0070913/example-ml-project) for the original version of this tutorial

Thank you to [Nikolai Janakiev](https://janakiev.com/blog/pytorch-iris/) for the toy model
