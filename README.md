# TransferLearningSample
My predictive model utilizes a deep learning architecture based on the Resnet model. I capitalize on the efficiency of a pre-trained Resnet model, fine-tuning it for the specific task of distinguishing between images of ants and bees. To streamline cross-system compatibility, I encapsulate the model using Docker. During the Docker image building process, all necessary requirements are automatically installed, and the dataset, comprising images of bees and ants distributed across training and validation folders, is seamlessly downloaded. Once the Docker image is constructed, users only need to execute the commands for training and testing. This simplifies the entire process, requiring minimal user intervention beyond initiating these essential steps.

1- Install Docker, VSCode and Remote-Containers extension  
2- Get the latest code: ```git clone https://github.com/Hamoon1987/TransferLearningSample.git```  
3- Go to the folder: ```cd TransferLearningSample```  
4- Build the docker image: ```docker image build -t tl .```  
5- Run the docker image: ```docker run -it -d --name my_tl tl```  
6- Attach to the running container using Remote-Containers extension  
7- You can see the dataloader details and a sample figure by running: ```python3 dataloader.py```  
8- Train the model: ```python3 train.py```  
9- Put the test image in the test folder and run: ```python3 test.py```  

