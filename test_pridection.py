from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('model.h5')

img_path = "cat1.jpg"

img = image.load_img(
    img_path,
    target_size=(128, 128)
)

img_array = image.img_to_array(img)
img_array = img_array / 255.0     # same normalization
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
if predictions[0] < 0.3:
    print("Predicted: Cat")
else:
    print("Predicted: Dog") 

# Note: Ensure that 'test_image.jpg' exists in the working directory for this code to run successfully.