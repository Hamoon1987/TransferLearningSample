# TransferLearningSample
My predictive model is a deep learning model based on Resnet architecture. I use the pre-trained Resnet model and fine-tune it to distinguish ants' and bees’ images. Also, I use docker to capsulate the model and make running the code on different systems easier. All the requirements are installed automatically while building the Docker image and the dataset which includes some images of bees and ants in training and validation folders are downloaded. After buidling the Docker image, running and attaching to it all user has to do is to run the training and testing.

1- Install Docker, VSCode and Remote-Containers extension  
2- Get the latest code: ```git clone https://github.com/Hamoon1987/TransferLearningSample.git``` 
3- Go to the folder: ```cd TransferLearningSample``` 
4- Build the docker image: ```docker image build -t tl .```  
5- Run the docker image: ```docker run -it -d --name my_tl tl```  
6- Attach to the running container using Remote-Containers extension 
7- You can see the dataloader details and a sample figure by running: ```python3 dataloader.py``` 
7- Train the model: python3 train.py  
8- Put the test image in the test folder and run: python3 test.py  

