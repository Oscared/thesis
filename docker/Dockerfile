FROM airlab404/dl:cuda10_pytorch_py36

# Install extras
COPY requirements.yml /requirements.yml
# If you are using a py27 image, change this to py27
RUN /bin/bash -c ". activate py36 && conda env update -f=/requirements.yml"
RUN /bin/bash -c ". activate py36 && conda install -y pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch"
CMD ["bash"]

WORKDIR /exp

