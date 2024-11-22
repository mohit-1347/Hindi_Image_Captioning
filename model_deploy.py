from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model without the optimizer
capmodel = load_model("best_model.h5", compile=False)

# Compile the model again with a valid optimizer
from tensorflow.keras.optimizers import Adam
capmodel.compile(optimizer=Adam(), loss="categorical_crossentropy")

# Paths to dataset
# dataset_path = "Flickr8k_Dataset"
captions_path = "hindi_captions.txt"

# Load captions
# Load captions and normalize keys
def load_captions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        captions = file.readlines()
    captions_dict = {}
    for line in captions:
        image, caption = line.split('\t')
        # Normalize image key by removing suffixes like #4
        image = image.split('#')[0].strip()
        caption = caption.strip()
        caption = f"<start> {caption} <end>"
        if image not in captions_dict:
            captions_dict[image] = []
        captions_dict[image].append(caption)
    return captions_dict

captions_dict = load_captions(captions_path)


all_captions = [cap for caps in captions_dict.values() for cap in caps]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

def extract_features(img_path):
    model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    # features = {}
    img = load_img(img_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img)


from PIL import Image
import matplotlib.pyplot as plt

def preview_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Input Image")
    plt.show()

max_length = 50

def generate_caption(model, tokenizer, features, max_length):
    caption = "<start>"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        prediction = model.predict([features, sequence], verbose=0)
        predicted_index = np.argmax(prediction)

        # Handle missing keys
        predicted_word = tokenizer.index_word.get(predicted_index, "<unknown>")

        if predicted_word == "end":
            break
        caption += " " + predicted_word
    return caption

# Test on an image
test_image = "blind.jpg"
preview_image(test_image)
test_features = extract_features(test_image)
caption = generate_caption(capmodel, tokenizer, test_features, max_length)
print("Generated Caption:", caption)
