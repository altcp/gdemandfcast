# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import io
import time
import yaml
import telepot
import requests
import picamera
import warnings
import numpy as np
import PIL.Image as Image

from os import walk

from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC


# %%
#Load Configuration 
with open('./config/config.yaml') as file:
    config = yaml.safe_load(file)


# %%
#Reproducibility
ID = config['settings']['random_seed_ID']
random.seed(ID)
np.random.seed(ID)
os.environ['PYTHONHASHSEED']=str(ID)
#warnings.filterwarnings("ignore")


# %%



# %%
#Load Model(s)
model = load_model(config['locations']['model_path'])
global_detector = MTCNN()


# %%
#Ultilty Function
def convert_to_embedding(face, model):

  face = face.astype('float32')
  mean, std = face.mean(), face.std()
  face = (face - mean) / std

  face_expanded = np.expand_dims(face, axis = 0)
  face_embedding = model.predict(face_expanded)

  return face_embedding


# %%
#Ultilty Function
def load_datatset(directory, model, required_size):

    X_list, y_list = list(), list()

    for subdir in os.listdir(directory):
        folder_path = directory + subdir + '/'
        (_, _, filenames) = next(os.walk(folder_path))
        #print(subdir)

        for name in filenames:
        #print(name)
        namepath = folder_path + name

        # read the image
        pixels = Image.open(namepath).convert('RGB')
        pixels = np.array(pixels)
        # detect faces
        faces = global_detector.detect_faces(pixels)

        num_of_face = len(faces)
        #print(num_of_face)

        for index, face in zip(range(len(faces)), faces):        
        # extract the face
            try:
                x, y, width, height = face['box']
                x = abs(x)
                y = abs(y)

                x2, y2 = x + width, y + height
                extracted_face = pixels[y:y2, x:x2]

                #print(index)  
                if (index > 0):
            
                    print("Duplicate Extraction Found In " + str(subdir) + ", File: " + str(namepath) + " : ")
                    #Debug
                    #plt.imshow(extracted_face)
                    #plt.show()
                    print(" ")
                    print(" ")
          
                else:

                    face_data = Image.fromarray(extracted_face)
                    face_data_resized = face_data.resize(required_size)

                    #Convert to Embedding
                    face_data_resized_np = np.asarray(face_data_resized)
                    face_embeded = convert_to_embedding(face_data_resized_np, model)

                    #Append to List
                    X_list.append(face_embeded)
                    y_list.append(subdir)

            except:
                print("Cannot Extract Face in" + str(subdir) + ", File: " + str(namepath))
                print(" ")
                print(" ")
                
  return X_list, y_list


# %%
def run_model(picture_taken):

    #Default
    clean_up()
    allow_or_disallow_entry = picture_taken

    training_dir = config['locations']['fit_path']
    X_train_list, y_train_list = load_datatset(training_dir, model, required_size = (160, 160))

    test_dir = config['locations']['test_path']
    X_test_list, y_test_list = load_datatset(test_dir, model, required_size = (160, 160))

    # Debug
    print("Type of X_test_list: ", type(X_test_list))
    print("Size of X_test_list: ", len(X_test_list))
    print(" ")
    print("Type of y_test_list: ", type(y_test_list))
    print("Size of y_test_list: ", len(y_test_list))

    X_train_num = np.asarray(X_train_list)
    X_test_num = np.asarray(X_test_list)

    #Debug
    print("Size of X_train: ", X_train_num.shape)
    print("Size of X_test: ", X_test_num.shape)

    encoder = LabelEncoder()
    encoder.fit(y_train_list)

    y_train_num = encoder.transform(y_train_list)
    y_test_num = encoder.transform(y_test_list)

    #Debug
    print("Size of y_train_num: ", y_train_num.shape)
    print("Size of y_test_num: ", y_test_num.shape)
    
    #Invoke Classifier
    preprocessor = Normalizer(norm='l2')
    norm_x_train = preprocessor.transform(X_train_num)
    norm_x_test = preprocessor.transform(X_test_num)

    clf = SVC(kernel='linear')
    clf.fit(norm_x_train, y_train_num)
    y_pred = clf.predict(norm_x_test)
    score = accuracy_score(y_test_num, y_pred)

    #Use accuracy score to deliver Automation
    min_score = config['configuration']['min_accuracy_for_entry'] 

    if (score > min_score):
        allow_or_disallow_entry = inform_property_owner(score)
        print("ADOS, Matched: {} % ".format(score))

        if (allow_or_disallow_entry == False):
             print("ADOS: Sorry, you are not authorised. ")

        else:
            print("ADOS: Welcome! The door is now opened for you.")

    else:
        print("ADOS: Sorry, you are not authorised to enter this Place. ")
    
    return True


# %%
def run_camera():

    try:

        with picamera.PiCamera() as camera:
        
            camera.resolution = (640, 480)
            camera.start_preview()

            # Camera warm-up time
            time.sleep(2)
            file_path = config['locations']['test_path'] 

            for i in range(5):
            
                filename = file_path + 'test_' + str(i) + '.jpg'
                camera.capture(filename)
                #Camera Shot Intervals
                time.sleep(1)
        
        return True    
    
    except:
        #Debug
        print("ADOS: Camera Failed to Capture.")

        return False
        


# %%
#Intergration Function 
def inform_property_owner(score):
    
    bot_token = config['api']['telegram']['token']
    bot_chatID = config['api']['telegram']['chat_ID']
    msg_one = "ADOS: Seeking Entry to Your Propery. Please kindly Verify. "
    msg_two = "ADOS: Enter 1 to allow entry or 2 to refuse entry. "
    
    #Default until Chat Bot Response
    response = 0

    bot = telepot.Bot(bot_token)
    bot.sendMessage(bot_chatID, msg_one)

    #Send Images
    file_path = config['locations']['test_path'] 

            for i in range(5):
                filename = file_path + 'test_' + str(i) + '.jpg'
                bot.sendPhoto(bot_chatID, filename)
        
    #Get the Decision.
    #Terence, are you able to help ?

    if(response == 1):
        decision = True
    else:
        decision = False

    return decision


# %%
def clean_up():

    try:
        
        folder_path = config['locations']['test_path'] 
        images_in_folder = os.listdir(folder_path)

        for image in images_in_folder:
            path_to_file = os.path.join(folder_path, image)
            os.remove(path_to_file)
    
    except:
        print("ADOS: No Image(s) Cleared. ")

    return None


# %%



