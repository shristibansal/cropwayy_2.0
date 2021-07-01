# init a base image (Alpine is small Linux distro)
FROM python:3.8.8-slim
# define the present working directory
WORKDIR /Cropwayy-Project
# copy the contents into the working dir
ADD . /Cropwayy-Project
# run pip to install the dependencies of the flask app
#RUN python3 -m pip install --upgrade https://files.pythonhosted.org/packages/64/d5/6ae9b85deabf09295c8e622862accbd1b718289d623544ea45151481ec1e/tensorflow_cpu-2.3.1-cp38-cp38-win_amd64.whl
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-cpu==2.3.1
RUN pip3 install -r requirements.txt
# define the command to start the container
CMD ["python","app.py"]