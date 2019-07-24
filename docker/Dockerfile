FROM cannici/dl:cuda10_pytorch1.1.0_py36

MAINTAINER Marco Cannici "marco.cannici@polimi.it"

# Use this for starting Jupyter Notebook using docker
# run-docker 3 0-5 jupyter notebook --ip=0.0.0.0 --no-browser 

# Install extras

COPY requirements.yml /requirements.yml

RUN /bin/bash -c ". activate py36 && conda env update -f=/requirements.yml"

 

# Enable jupyter

RUN mkdir /.local

RUN chmod -R 777 /.local

CMD ["bash"]

 

WORKDIR /exp
