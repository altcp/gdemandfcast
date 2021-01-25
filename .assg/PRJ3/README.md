### ARIP2 - PRJ3 Automated Dook Opening System ###


Lionel 
Terence 
Tan Chiang Pern Alvin, s99661239



### Overview ###


This project was done in fufilment of the assignment three (3) conditions for ARIP 2. The project centered upon real-life applications of machine learning. In our application of artifical intelligence, we decided explore the implementation for an Automated Dock Opening System using what learnt in DLAE. In doing so, we implemented the key functionalities of such a system.  


Specifically, we implemented 1) the image capture using the python module, picamera and the offical Pi Camera Board, 2) the AI facial verification using MTCNN and Pre-Trained Facenet Model learnt in DLAE, 3) the delivery of such automation using the Telegram Bot.   



### Structure of Base Folder (ARIP2-PRJ3) ###


|
| - config
|        -- config.yml
|
|
| - data
|      -- pre_trained_model           
|                         --- facenet_keras.h5
|      -- test_data
|      -- train_data   
|                  --- alvin
|                  --- terance
|                  --- lionel
|
|
| - src
|     -- main.py
|     -- functions.py
|
|
| - README.md
| - AA.ipynb
| - requirements.txt
|
| 
| - run.sh
|



### Instructions for Use ###


Using our supplied Raspberry PI, SSH in and TYPE: bash run.sh


To inspect the source code in jupyter lab, navigate to the base folder and launch the jupyter notebooks, main.ipynb and functions.ipynb. 

You will need to install dependencies using the requirements.txt in our base folder.
Navigate to our base folder in a Linux Terminal or the equivalent in a Windows Subsystem for Linux, and TYPE:
pip install -r requirements.txt (assuming you have PIP package already installed.)  

To configure Hyperparameters, navigate to the config folder and edit the config.yml file - Instructions Within.To add yourself to the automation, please include a sub folder with your photos in ./data/train_dat/[yourname]/



### Explainations and Evaluations ###


We decided to use the pre-trained model for pragmatic reasons. Our intent is show that we are able to apply what we learnt in real life application(s) and that we can execute what we learn in this course. To be able to ship a product is as important as knowing how to code an AI model. Accordingly, our focus is on the former given that we are trying to put AI into Engineering which we feel is the crux of this project assigned to us, academically. 



### Work Distribution ###


Alvin worked on the APP Logic, Packaging and Documentation.
Terence worked on the Chatbot Integration and OS Install.
Lionel worked on the Hardware and Electronics Integration.

The Team tested the APP together.


