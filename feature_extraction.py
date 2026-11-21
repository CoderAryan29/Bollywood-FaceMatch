#WE CREATED A FILENAMES.PKL WITH EVERY DATA OF ACTORS AND ACTRESSES IN IT FROM THIS CODE AND THEN WE COMMENTED IT OUT
'''import os
import pickle

base_path = 'data/facematcher data/Bollywood_celeb_face_localized'

filenames = []

# All three dataset folders
dataset_folders = [
    'bollywood_celeb_faces2',
    'bollywood_celeb_faces_0',
    'bollywood_celeb_faces_1'
]

# Collect image paths from every celebrity folder
for dataset in dataset_folders:
    dataset_path = os.path.join(base_path, dataset)

    for actor in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            file_path = os.path.join(actor_path, file)

            if (
                os.path.isfile(file_path)
                and file.lower().endswith(('.jpg', '.jpeg', '.png'))
            ):
                filenames.append(file_path)

print("Total images collected:", len(filenames))

# Save the paths into a pickle file
with open('filenames.pkl', 'wb') as file:
    pickle.dump(filenames, file)

print("All filenames saved to filenames.pkl")
'''

from tensorflow.keras.preprocessing import image  
from keras_vggface.utils import preprocess_input  
from keras_vggface.vggface import VGGFace  
import numpy as np  
import pickle  
from tqdm import tqdm
'''
image	            Loads and converts image files
preprocess_input	Prepares image pixel values for VGGFace
VGGFace	            Loads the pretrained face recognition model
numpy	            Performs numerical and array operations
pickle	            Saves and loads Python objects
tqdm	            Displays a progress bar
''' 

filenames = pickle.load(open('filenames.pkl', 'rb'))

model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224,224,3),
    pooling='avg'
)


def feature_extractor(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img)

    result=model.predict(preprocessed_img).flatten()

    return result

features=[]

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embeddings.pkl','wb'))
   


