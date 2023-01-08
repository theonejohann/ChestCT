from tkinter import *
from tkinter import filedialog
import numpy as np
import tensorflow as tf

class_names = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 'normal', 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']



# Load the H5 model
model = tf.keras.models.load_model("model.h5")

root = Tk()

# Create a function to open the file browser and select an image file
def open_image():
  # Open the file browser
  filepath = filedialog.askopenfilename()

  # Create an image object from the selected file
  my_image = PhotoImage(file=filepath)
  # Create a label widget with the image
  label_image = Label(root, image=my_image)
  label_image.image = my_image
  # Pack the label into the window
  label_image.pack()

  # Preprocess the image
  img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  label = Label(root, text="This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score), font=("Arial", 20 )))
  label.pack()
  

  
    

# Create a button to trigger the file open dialog
button = Button(root, text="Open Image", command=open_image)

label = Label(root, text="CT Scanner", font=("Arial", 24), width=20, height=2)

label.pack()

# Pack the button into the window
button.pack()

root.mainloop()
