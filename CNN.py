import numpy as np
import argparse
import cv2
from CNN_Keras.cnn.neural_network import CNN
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# Parse the Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1)
ap.add_argument("-l", "--load_model", type=int, default=-1)
ap.add_argument("-w", "--save_weights", type=str)
args = vars(ap.parse_args())

# Read/Download dataset from the directory on the computer

print('Loading Dataset.')
data1 = loadImages(r"C:\Users\ADMIN\Desktop\Test\Data\Train\East")
data2 = loadImages(r"C:\Users\ADMIN\Desktop\Test\Data\Train\West")
data3 = loadImages(r"C:\Users\ADMIN\Desktop\Test\Data\Train\South")
data4 = loadImages(r"C:\Users\ADMIN\Desktop\Test\Data\Train\North")
test_data = loadImages(r"C:\Users\ADMIN\Desktop\Test\Data\Test")
train_data = [data1, data2, data3, data4]

# Divide data into testing and training sets.
train_img, test_img, train_labels, test_labels = train_test_split(train_data, test_data, test_size=0.1)

# Now each image rows and columns are of 28x28 matrix type.
img_rows, img_columns = 100, 100

# Transform training and testing data to 10 classes in range [0,classes] ; num. of classes = 0 to 9 = 10 classes
total_classes = 4  # 0 to 9 labels
train_labels = np_utils.to_categorical(train_labels, 4)
test_labels = np_utils.to_categorical(test_labels, 4)

# Defining and compile the SGD optimizer and CNN model
print('\n Compiling model...')
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
clf = CNN.build(width=100, height=100, depth=1, total_classes=4,
                Saved_Weights_Path=args["save_weights"] if args["load_model"] > 0 else None)
clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Initially train and test the model; If weight saved already, load the weights using arguments.
b_size = 40  # Batch size
num_epoch = 10  # Number of epochs
verb = 1  # Verbose

# If weights saved and argument load_model; Load the pre-trained model.
if args["load_model"] < 0:
    print('\nTraining the Model...')
    clf.fit(train_img, train_labels, batch_size=b_size, epochs=num_epoch, verbose=verb)

    # Evaluate accuracy and loss function of test data
    print('Evaluating Accuracy and Loss Function...')
    loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
    print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

# Save the pre-trained model.
if args["save_model"] > 0:
    print('Saving weights to file...')
    clf.save_weights(args["save_weights"], overwrite=True)

# Show the images using OpenCV and making random selections.
for num in np.random.choice(np.arange(0, len(test_labels)), size=(5,)):
    # Predict the label using CNN.
    probs = clf.predict(test_img[np.newaxis, num])
    prediction = probs.argmax(axis=1)

    # Resize the Image to 100x100 from 28x28 for better view.

    image = (test_img[num][0] * 255).astype("uint8")
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (1000, 1000), interpolation=cv2.INTER_LINEAR)

    #Bounding Box

    def draw_rects(img, rects):

       for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 00), 2)
        object = img
        cv2.putText(img, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Show and print the Actual Image and Predicted Label Value
    print('Predicted Label: {}, Actual Value: {}'.format(prediction[0], np.argmax(test_labels[num])))
    cv2.imshow('Direction', image)
    cv2.waitKey(0)


