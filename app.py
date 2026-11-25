import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import types
import pickle
import numpy as np
from PIL import Image
import keras.utils

# Compatibility shims for Keras / Python 3.11+
try:
    from keras.utils import layer_utils
except ImportError:
    keras.utils.layer_utils = types.ModuleType('layer_utils')
    sys.modules['keras.utils.layer_utils'] = keras.utils.layer_utils

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
from mtcnn import MTCNN
import gradio as gr

# Required by Hugging Face ZeroGPU startup scanner
try:
    import spaces
    @spaces.GPU
    def _zero_gpu_heartbeat():
        pass
except ImportError:
    pass

# Initialize models
detector = MTCNN()
model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)

# Load precomputed embeddings and normalize once for instant ~3ms matrix dot-product
feature_list_raw = np.array(pickle.load(open('embeddings.pkl', 'rb')), dtype=np.float32)
norms = np.linalg.norm(feature_list_raw, axis=1, keepdims=True)
norms[norms == 0] = 1e-10
feature_matrix = feature_list_raw / norms

filenames = pickle.load(open('filenames.pkl', 'rb'))


def match_face(input_image):
    if input_image is None:
        return None, "⚠️ **Please upload or take a photo first!**"

    # Downscale photo to max 800px so face detection is snappy
    input_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
    img = np.array(input_image.convert('RGB'))

    # Detect faces with MTCNN
    results = detector.detect_faces(img)
    if len(results) == 0:
        return None, "⚠️ **No face detected.** Please upload a clear photo with a visible face."

    # Pick largest detected face
    results = sorted(
        results,
        key=lambda face: face['box'][2] * face['box'][3],
        reverse=True
    )
    x, y, width, height = results[0]['box']

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img.shape[1], x + width)
    y2 = min(img.shape[0], y + height)

    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None, "⚠️ **Could not crop face.** Please try another photo."

    # Resize and preprocess for VGGFace ResNet50
    face_pil = Image.fromarray(face).resize((224, 224))
    face_array = np.asarray(face_pil).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    # Extract 2048-dim feature embedding
    features = model.predict(preprocessed_img, verbose=0).flatten().astype(np.float32)

    # Instant vector dot-product matching in ~3ms
    feat_norm = features / (np.linalg.norm(features) + 1e-10)
    similarities = np.dot(feature_matrix, feat_norm)

    best_idx = int(np.argmax(similarities))

    # Lookup matched celebrity
    matched_path = filenames[best_idx].replace('\\', '/')
    actor_name = os.path.basename(os.path.dirname(matched_path)).replace('_', ' ')

    matched_image = None
    if os.path.exists(matched_path):
        matched_image = Image.open(matched_path).convert('RGB')

    # Display clean celebrity name without confidence score
    result_text = f"## 🌟 Seems like **{actor_name}**!"

    return matched_image, result_text


# Build Gradio UI with balanced, prominent image containers
with gr.Blocks(title="Bollywood FaceMatch") as demo:
    gr.Markdown(
        """
        # 🎬 Bollywood FaceMatch
        ### Discover Your Bollywood Celebrity Lookalike with Deep Facial Embeddings!
        Upload any clear picture of a face, and our deep learning model (**MTCNN + VGGFace ResNet50**) will find your closest Bollywood celebrity match.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Your Photo (Upload or Selfie)", height=500)
            match_btn = gr.Button("🔍 Find My Bollywood Match", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="Matched Bollywood Celebrity", height=500)
            output_text = gr.Markdown("### 👈 Upload a photo and click the button to see your match!")

    match_btn.click(
        fn=match_face,
        inputs=input_img,
        outputs=[output_img, output_text]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
