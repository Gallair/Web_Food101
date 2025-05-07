from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)
model = load_model('101_food_classes_10_percent_saved_big_dog_model.keras')
target_img = os.path.join(os.getcwd(), 'static/images')


@app.route('/')
def index_view():
    return render_template('index2.html')


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT


# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_and_prep_image(filename, img_shape=224, scale=True):
    """
      Reads in an image from filename and turns it into a Tensor and reshapes it to (img_shape, img_shape, color_channels = 3)

      Args:
        filename (str): path to target image
        img_shape (int): height/weight dimension of target image, default 224
        scale (bool): whether to scale pixel values from (0, 255) to range(0, 1), default True

      Returns:
        Tensor of shape (img_shape, img_shape, 3)
    """
    # Read in the image file
    img = tf.io.read_file(filename)

    # Decode image into tensor
    img = tf.image.decode_image(img, channels=3)

    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])

    # Scale? Yes/No
    if scale:
        # rescale the image (get all values between 0 and 1)
        return img / .255
    else:
        return img  # don't need to rescale images for EffcientNet models in TensorFlow

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', str(filename))
            fp = 'static/images' + "/" + str(filename)
            file.save(fp)
            image_my = fp
            #image_my = 'static/images' + "/" + str("20250403_061846.jpg")
            #img = read_image(image_my)  # prepressing method
            #class_prediction = model.predict(img)
            img = load_and_prep_image(image_my, scale = False)

            class_prediction = model.predict(tf.expand_dims(img, axis = 0))

            class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio',
             'beef_tartare', 'beet_salad',  'beignets',  'bibimbap', 'bread_pudding',
             'breakfast_burrito',  'bruschetta',  'caesar_salad', 'cannoli', 'caprese_salad',
             'carrot_cake', 'ceviche',  'cheese_plate', 'cheesecake', 'chicken_curry',
             'chicken_quesadilla',  'chicken_wings',   'chocolate_cake',  'chocolate_mousse',
             'churros', 'clam_chowder',  'club_sandwich', 'crab_cakes', 'creme_brulee',
             'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts',  'dumplings',
             'edamame',   'eggs_benedict',  'escargots',  'falafel',  'filet_mignon',
             'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup',
             'french_toast',  'fried_calamari',  'fried_rice',  'frozen_yogurt',
             'garlic_bread', 'gnocchi', 'greek_salad',  'grilled_cheese_sandwich',
             'grilled_salmon',  'guacamole', 'gyoza',  'hamburger',  'hot_and_sour_soup',
             'hot_dog', 'huevos_rancheros',  'hummus', 'ice_cream', 'lasagna',
             'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons',
             'miso_soup', 'mussels',  'nachos', 'omelette', 'onion_rings',
             'oysters','pad_thai', 'paella','pancakes', 'panna_cotta', 'peking_duck',
             'pho', 'pizza', 'pork_chop', 'poutine',  'prime_rib', 'pulled_pork_sandwich',
             'ramen',  'ravioli', 'red_velvet_cake',  'risotto','samosa', 'sashimi',
             'scallops',  'seaweed_salad',  'shrimp_and_grits', 'spaghetti_bolognese',
             'spaghetti_carbonara', 'spring_rolls','steak','strawberry_shortcake',  'sushi',
             'tacos','takoyaki',  'tiramisu',  'tuna_tartare', 'waffles']

            pred_class = class_names[class_prediction.argmax()]

            return render_template('predict2.html', food=pred_class, prob=class_prediction.max(), user_image=image_my)
        else:
            return "Unable to read the file. Please check file extension"
    else:
        return "no POST"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
