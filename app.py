import sys
import types
import keras.utils

# 1. Fix missing layer_utils (for newer Keras on Python 3.11+)
try:
    from keras.utils import layer_utils
except ImportError:
    keras.utils.layer_utils = types.ModuleType('layer_utils')
    sys.modules['keras.utils.layer_utils'] = keras.utils.layer_utils

# 2. Fix missing data_utils
try:
    from keras.utils.data_utils import get_file
except ImportError:
    data_utils = types.ModuleType('data_utils')
    try:
        from tensorflow.keras.utils import get_file
    except ImportError:
        from keras.utils import get_file
    data_utils.get_file = get_file
    keras.utils.data_utils = data_utils
    sys.modules['keras.utils.data_utils'] = data_utils

# 3. Fix missing keras.engine.topology
try:
    from keras.engine.topology import get_source_inputs
except ImportError:
    topology = types.ModuleType('topology')
    try:
        from tensorflow.keras.utils import get_source_inputs
    except ImportError:
        from keras.utils import get_source_inputs
    topology.get_source_inputs = get_source_inputs
    sys.modules['keras.engine.topology'] = topology

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

import pickle
import streamlit as st
from PIL import Image
import os
import io
import base64
import cv2
from mtcnn import MTCNN
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def image_to_data_uri(pil_img, max_size=800):
    """Convert a PIL image to an inline base64 JPEG data URI so st.image bypasses MediaFileManager."""
    img_copy = pil_img.copy()
    img_copy.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img_copy.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


# -------------------------------
# Streamlit page
# -------------------------------

st.set_page_config(
    page_title="Bollywood FaceMatch",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Which Bollywood Celebrity Are You?")


# -------------------------------
# Load model and saved data
# -------------------------------

@st.cache_resource
def load_model_data():

    detector = MTCNN()

    model = VGGFace(
        model='resnet50',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )

    feature_list = pickle.load(
        open('embeddings.pkl', 'rb')
    )

    filenames = pickle.load(
        open('filenames.pkl', 'rb')
    )

    return detector, model, feature_list, filenames


detector, model, feature_list, filenames = load_model_data()


# -------------------------------
# Save uploaded image
# -------------------------------

def save_uploaded_image(uploaded_image):

    try:
        os.makedirs('uploads', exist_ok=True)

        image_path = os.path.join(
            'uploads',
            uploaded_image.name
        )

        with open(image_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())

        return image_path

    except Exception as e:
        st.error(f"Could not save image: {e}")
        return None


# -------------------------------
# Extract face features
# -------------------------------

def extract_features(img_path, model, detector):

    img = cv2.imread(img_path)

    if img is None:
        return None

    # OpenCV BGR to RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Detect face
    results = detector.detect_faces(img)

    if len(results) == 0:
        return None

    # Select the largest detected face
    results = sorted(
        results,
        key=lambda face: face['box'][2] * face['box'][3],
        reverse=True
    )

    x, y, width, height = results[0]['box']

    # Clip coordinates to image boundaries
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img.shape[1], x + width)
    y2 = min(img.shape[0], y + height)

    # Crop face
    face = img[y1:y2, x1:x2]

    if face.size == 0:
        return None

    # Resize face
    face = Image.fromarray(face)
    face = face.resize((224, 224))

    # Convert to NumPy array
    face_array = np.asarray(face).astype('float32')

    # Add batch dimension
    expanded_img = np.expand_dims(
        face_array,
        axis=0
    )

    # Preprocess for VGGFace
    preprocessed_img = preprocess_input(expanded_img)

    # Extract embedding
    result = model.predict(
        preprocessed_img,
        verbose=0
    ).flatten()

    return result


# -------------------------------
# Find closest celebrity
# -------------------------------

def recommend(feature_list, features):

    similarity = []

    for i in range(len(feature_list)):

        score = cosine_similarity(
            features.reshape(1, -1),
            feature_list[i].reshape(1, -1)
        )[0][0]

        similarity.append(score)

    # Index of highest similarity
    index_pos = sorted(
        enumerate(similarity),
        reverse=True,
        key=lambda x: x[1]
    )[0][0]

    return index_pos


# -------------------------------
# Upload image
# -------------------------------

uploaded_image = st.file_uploader(
    "Upload your image",
    type=['jpg', 'jpeg', 'png', 'webp']
)


# -------------------------------
# Process uploaded image
# -------------------------------

if uploaded_image is not None:

    uploaded_image.seek(0)
    display_image = Image.open(
        uploaded_image
    ).convert('RGB')
    display_uri = image_to_data_uri(display_image)

    st.image(
        display_uri,
        caption="Your uploaded image",
        width=300
    )

    if st.button("Find My Bollywood Match", type="primary"):

        image_path = save_uploaded_image(uploaded_image)

        if image_path is not None:

            with st.spinner("Finding your Bollywood match..."):

                features = extract_features(
                    image_path,
                    model,
                    detector
                )

            if features is None:

                st.error(
                    "No face detected. Please upload a clear face image."
                )

            else:

                # Find best match
                index_pos = recommend(
                    feature_list,
                    features
                )

                # Convert Windows '\' to '/' for cross-platform compatibility
                matched_path = filenames[index_pos].replace('\\', '/')

                # Get actor name from parent folder
                predicted_actor = os.path.basename(
                    os.path.dirname(matched_path)
                ).replace('_', ' ')

                # Load matched celebrity image
                matched_image = Image.open(
                    matched_path
                ).convert('RGB')

                # Display result
                st.success("Match found!")

                col1, col2 = st.columns(2)

                with col1:

                    st.header("Your Uploaded Image")

                    st.image(
                        display_uri,
                        use_container_width=True
                    )

                with col2:

                    st.header(
                        f"Seems like {predicted_actor}!"
                    )

                    st.image(
                        image_to_data_uri(matched_image),
                        use_container_width=True
                    )
