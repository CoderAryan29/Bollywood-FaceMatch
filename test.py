#load image --> face detect and extract its features
#find the cosine distance of current image with all the features 
#recommend the closest image

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input  
from keras_vggface.vggface import VGGFace  
import numpy as np  
import pickle  
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image


feature_list=np.array(pickle.load(open('embeddings.pkl',"rb")))
filenames=pickle.load(open('filenames.pkl',"rb"))

model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224,224,3),
    pooling='avg'
)

detector=MTCNN()
#load image --> face detect
sample_img = cv2.imread('sample/33.webp')
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
result = detector.detect_faces(sample_img)

x, y, width, height = result[0]['box']
x1, y1 = max(0, x), max(0, y)
x2, y2 = min(sample_img.shape[1], x + width), min(sample_img.shape[0], y + height)

face = sample_img[y1:y2, x1:x2]

image=Image.fromarray(face)
image=image.resize((224,224))

face_array=np.asarray(image)
face_array=face_array.astype('float32')
expanded_img = np.expand_dims(face_array, axis=0)

preprocessed_img=preprocess_input(expanded_img)

result=model.predict(preprocessed_img).flatten()

similarity=[]
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
temp_img=cv2.imread(filenames[index_pos[0][0]])
cv2.imshow('output',temp_img)
cv2.waitKey(0)