import tensorflow.keras.layers as tfl
from tensorflow.keras import Model, Input, utils, regularizers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow import reshape
from keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.data import Dataset
import h5py
import tensorflow as tf
import numpy as np


def backbone_model(
    image_shape = (224, 224 , 3),
    category_depth = 9,
    trainable = False):

    backbone = MobileNetV3Small(
    input_shape= image_shape,
    include_top= False,
    weights= 'imagenet',
    input_tensor= None,
    dropout_rate= 0.2,
    include_preprocessing= True)

    backbone.trainable = trainable

    last_layer = backbone.get_layer(backbone.layers[-1].name).output

    conv_output = backbone.get_layer(f'expanded_conv_{str(category_depth)}/Add').output

    return Model(inputs = backbone.input, outputs = [last_layer, conv_output])

def flat_model(
    image_shape = (224, 224 , 3),
    num_class = 4,
    L2 = 0
):
    backbone = backbone_model(
    image_shape = image_shape,
    trainable = False)

    backbone.trainable = False

    input = Input(shape = image_shape)

    [last_layer, conv_output] = backbone(input, training = False)

    X = tfl.Flatten()(last_layer)
    X = tfl.Dense(200, activation = 'relu', kernel_regularizer= regularizers.l2(L2))(X)
    X = tfl.Dropout(0.2)(X)
    X = tfl.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2))(X)
    X = tfl.Dropout(0.2)(X)
    outputs = tfl.Dense(num_class, activation = 'softmax', name = 'category')(X)

    model = Model(inputs = input, outputs = outputs)

    return model

file_path = 'hierachical_5_3_224x224_30350.h5'
# file_path = '4label_flat.h5'

with h5py.File(file_path, 'r') as hf:
    X = hf['images'][()]
    Class = hf['class'][()]
print(X.shape)
X_shape = X.shape
print(Class.shape)
num_class = len(set(Class))
print(num_class)
Labels = np.array(reshape(utils.to_categorical(Class,num_class),(-1,num_class)))

print(Labels.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, Labels_train, Labels_test = train_test_split(
    X, Labels, test_size=0.2, random_state=42
)
del Class
del X
train_set = Dataset.from_tensor_slices((X_train, Labels_train))
train_set = train_set.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
train_set = train_set.batch(32)

test_set = Dataset.from_tensor_slices((X_test, Labels_test))
test_set = test_set.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
test_set = test_set.batch(32)

del X_train
del Labels_train
del Labels_test

lr_schedule = ExponentialDecay(
    2e-4,
    decay_steps=2500,
    decay_rate=0.7,
    staircase=True
)
train_set = train_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

cardinary_model = flat_model(image_shape=(X_shape[1],X_shape[2],3), num_class=num_class, L2=0.2)
cardinary_model.compile(optimizer = Adam(learning_rate=lr_schedule,), loss = 'categorical_crossentropy', metrics = ['accuracy'])
cardinary_model.fit(train_set, epochs = 45)




results = cardinary_model.evaluate(test_set)
print(results)
all_X = []
all_y = []
for batch in test_set:
    X, y = batch  # Unpack the batch into X (inputs) and y (labels)
    # Collect batch of inputs
    all_X.append(X.numpy())  # Assuming X is a TensorFlow tensor, convert to NumPy if needed
    all_y.append(y.numpy())  # Assuming y is a TensorFlow tensor, convert to NumPy if needed
# Concatenate all batches into single arrays or lists
all_X = tf.concat(all_X, axis=0)  # Concatenate along batch dimension if X is a tensor
all_y = tf.concat(all_y, axis=0)  # Concatenate along batch dimension if y is a tensor
def category_accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis = 1)//3
    y_pred = tf.argmax(y_pred, axis = 1)//3
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

def grade_accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis = 1).numpy()%3
    y_pred = tf.argmax(y_pred, axis = 1).numpy()%3
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

y_pred = cardinary_model.predict(all_X)
print('Category_Accuracy :', category_accuracy(all_y, y_pred).numpy())
print('Grade_Accuracy :', grade_accuracy(all_y, y_pred).numpy())