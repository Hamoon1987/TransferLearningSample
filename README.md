# TransferLearningSample
My predictive model is a deep learning model based on Resnet architecture. I use the pre-trained Resnet model and fine-tune it to distinguish ants' and bees’ images. Also, I use docker to capsulate the model and make running the code on different systems easier.

1- Login to Docker and Github \
2- Get the docker image: sudo docker pull hamoonjafarian/transfer_learning \
3- Create the container: sudo docker run -it -d --name my_tl hamoonjafarian/transfer_learning \
4- Execute the container: sudo docker exec -it my_tl bash \
5- Clone the repo: git clone -b version02 https://github.com/Hamoon1987/TransferLearningSample.git \
6- Get data: gdown https://drive.google.com/uc?id=1pvLxfABGHxlSRC4CPvRGQD_NdDlTISJ0 \
7- Extract data: tar -xf data.tar\
8- Train: python3 train.py \
9- Create test folder and put an image inside \
10- Test: python3 test.py
