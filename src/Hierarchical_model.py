import tensorflow.keras.layers as tfl
from tensorflow.keras import Model, Input, utils, regularizers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow import concat, reshape, GradientTape
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset
import h5py
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class HierarchicalModel(Model):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs)

    def set_hyper(self, lamb, sigma):
        self.lamb = lamb
        self.sigma = sigma

    def train_step(self, data):
        x, y = data
        y_true_category = y['category']
        y_true_grade = y['grade']
        with GradientTape() as tape:
            [y_pred_category, y_pred_grade] = self(x, training=True)
            category_loss = categorical_crossentropy(y_true_category, y_pred_category)
            grade_loss = categorical_crossentropy(y_true_grade, y_pred_grade)

            correct_category = tf.reduce_all(tf.equal(tf.argmax(y_true_category, axis=1), tf.argmax(y_pred_category, axis=1)), axis=-1)
            correct_grade = tf.reduce_all(tf.equal(tf.argmax(y_true_grade, axis=1), tf.argmax(y_pred_grade, axis=1)),axis=-1)
            both_correct = tf.cast(correct_category & correct_grade, tf.float32)
            penalty = 1.0 - both_correct  # 1 if either is wrong, 0 if both are correct

            combined_loss = (1-self.lamb)* category_loss + self.lamb* grade_loss + self.sigma * penalty * (category_loss*grade_loss)

        # Compute gradients
        gradients = tape.gradient(combined_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, [y_pred_category, y_pred_grade])

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

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


def hierarchical_model(
    image_shape = (224, 224 , 3),
    category_depth = 9,
    category_num = 2,
    grade_num = 2,
    L2 = 0):


    backbone = backbone_model(
    image_shape = image_shape,
    category_depth = category_depth,
    trainable = False)

    backbone.trainable = False

    input = Input(shape = image_shape)

    [x, conv_output] = backbone(input, training = False)

    x_category = tfl.Flatten()(conv_output)
    x_category = tfl.Dropout(0.4)(x_category)
    x_category = tfl.Dense(256, activation = 'relu', kernel_regularizer= regularizers.l2(L2))(x_category)
    x_category = tfl.Dropout(0.3)(x_category)
    x_category = tfl.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2))(x_category)
    x_category = tfl.Dropout(0.25)(x_category)
    category_output = tfl.Dense(category_num, activation = 'softmax', name = 'category', kernel_regularizer=regularizers.l2(L2))(x_category)


    x_grade = tfl.Flatten()(x)
    x_grade = tfl.Dropout(0.5)(x_grade)
    x_grade = tfl.Dense((100), activation='relu', kernel_regularizer=regularizers.l2(L2*3.5))(x_grade)
    x_grade = tfl.Dropout(0.35)(x_grade)
    x_grade = tfl.Dense((16-category_num), activation = 'relu', kernel_regularizer= regularizers.l2(L2*3.5))(x_grade)
    x_grade = concat([x_grade, category_output], -1)

    grade_output = tfl.Dense(grade_num, activation = 'softmax', name = 'grade', kernel_regularizer= regularizers.l2(L2*3.5))(x_grade)

    model = HierarchicalModel(inputs = input, outputs = [category_output, grade_output])

    return model

def Overall_accuracy(y_true, y_pred):
    # Unpack true labels
    y_true_category = y_true[0]
    y_true_grade = y_true[1]

    # Unpack predicted labels
    y_pred_category = y_pred[0]
    y_pred_grade = y_pred[1]

    # Get the predicted class (highest probability) for both outputs
    y_pred_category = tf.argmax(y_pred_category, axis=-1)
    y_pred_grade = tf.argmax(y_pred_grade, axis=-1)

    # Get the true class for both outputs
    y_true_category = tf.argmax(y_true_category, axis=-1)
    y_true_grade = tf.argmax(y_true_grade, axis=-1)

    # Compare both predictions with true labels
    is_correct_category = tf.equal(y_pred_category, y_true_category)
    is_correct_grade = tf.equal(y_pred_grade, y_true_grade)

    # Only count as correct if both category and grade are correct
    is_correct = tf.logical_and(is_correct_category, is_correct_grade)
    # Convert boolean tensor to float and take the mean to get accuracy
    return tf.reduce_mean(tf.cast(is_correct, tf.float32))

def Category_accuracy(y_true, y_pred):
    y_true_category = tf.argmax(y_true, axis=-1)
    y_pred_category = tf.argmax(y_pred, axis=-1)
    is_true = tf.equal(y_true_category, y_pred_category)
    return tf.reduce_mean(tf.cast(is_true, tf.float32))

def Grade_accuracy(y_true, y_pred):
    y_true_grade = tf.argmax(y_true, axis=-1)
    y_pred_grade = tf.argmax(y_pred, axis=-1)
    is_true = tf.equal(y_true_grade, y_pred_grade)
    return tf.reduce_mean(tf.cast(is_true, tf.float32))


if __name__ == '__main__':

    #loading data set
    tf.random.set_seed(42)
    file_path = 'Data\hierachical_5_3_224x224_6976.h5'
    # file_path = r'Data\4label.h5'

    with h5py.File(file_path, 'r') as hf:
        X = hf['images'][()]
        y_category = hf['labels'][()]
        y_grade = hf['qualities'][()]
        print('Category Labels:', set(y_category))
        print('Grade Labels:', set(y_grade))
    print("X's shape:", X.shape)
    print("Y_category's shape:", y_category.shape)
    print("Y_grade's shape:", y_grade.shape)
    category_num = len(set(y_category))
    grade_num = len(set(y_grade))
    input_shape = (X.shape[1], X.shape[2], 3)

    print('Number of categories:', category_num)
    print('Number of grades:', grade_num)

    #change labels to one-hot encoding
    y_category_onehot = np.array(reshape(utils.to_categorical(y_category,category_num),(-1,category_num)))
    y_grade_onehot = np.array(reshape(utils.to_categorical(y_grade,grade_num),(-1,grade_num)))
    #split data into training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_category_train, y_category_test, y_grade_train, y_grade_test = train_test_split(
    X, y_category_onehot, y_grade_onehot, test_size=0.2, random_state=42
    )

    #create dataset
    train_dataset = Dataset.from_tensor_slices((X_train, {'category': y_category_train, 'grade': y_grade_train}))
    train_dataset = train_dataset.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(32)
    test_dataset = Dataset.from_tensor_slices((X_test, {'category': y_category_test, 'grade': y_grade_test}))
    test_dataset = test_dataset.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
    test_dataset = test_dataset.batch(32)


    all_X = []
    all_y_category = []
    all_y_grade = []

    # Iterate over the dataset to collect all data and labels
    for batch in test_dataset:
        X, y = batch  # Unpack the batch into X (inputs) and y (labels)

        # Collect batch of inputs
        all_X.append(X.numpy())  # Assuming X is a TensorFlow tensor, convert to NumPy if needed

        # Collect batch of labels
        all_y_category.append(y['category'].numpy())  # Assuming y['category'] is a TensorFlow tensor
        all_y_grade.append(y['grade'].numpy())  # Assuming y['grade'] is a TensorFlow tensor

        # Concatenate all batches into single arrays or lists
    all_X = tf.concat(all_X, axis=0)  # Concatenate along batch dimension if X is a tensor
    all_y_category = tf.concat(all_y_category, axis=0)  # Concatenate along batch dimension if y['category'] is a tensor
    all_y_grade = tf.concat(all_y_grade, axis=0)  # Concatenate along batch dimension if y['grade'] is a tensor


    lr_schedule = ExponentialDecay(
    4e-5,
    decay_steps=3000,
    decay_rate=0.7,
    staircase=True
    )

    model = hierarchical_model(image_shape=input_shape, category_num=category_num, grade_num=grade_num, L2=0.35)

    model.load_weights('Hierarchical_weight.h5')
    model.compile(optimizer = Adam(learning_rate=lr_schedule),loss='categorical_crossentropy', metrics = ['accuracy'])#, lamb = 0.5, sigma = 5)
    # model.set_hyper(lamb= 0, sigma = 0)
    # history = model.fit(train_dataset, epochs =30)
    # with open('Multitask_Learning_Category_1.txt', 'w') as f:
    #     f.write(str(history.history))
    # lr_schedule = ExponentialDecay(
    #     8e-5,
    #     decay_steps=2100,
    #     decay_rate=0.7,
    #     staircase=True
    # )
    # model.compile(optimizer = Adam(learning_rate=lr_schedule),loss='categorical_crossentropy', metrics = ['accuracy'])
    # model.fit(train_dataset, epochs = 20)

    outputs = model.predict(all_X)
    [pred_category, pred_grade] = outputs
    results = model.evaluate(test_dataset)


    absolute_acc = Overall_accuracy([all_y_category,all_y_grade], [pred_category, pred_grade]).numpy()
    category_acc = Category_accuracy(all_y_category, pred_category).numpy()
    grade_acc = Grade_accuracy(all_y_grade, pred_grade).numpy()
    print(f'Absolute Accuracy: {absolute_acc} \n Category Accuracy: {category_acc} \n Grade Accuracy: {grade_acc}')
    # model.save_weights('Hierarchical_weight.h5')


    # result_list = []
    # for sigma in range(0,100,5):
    #     temp_abs = 0
    #     temp_cat = 0
    #     temp_grade = 0
    #     for i in range(5):
    #         model = hierarchical_model(image_shape=(x_train.shape[1],x_train.shape[2], 3), category_num=5, grade_num=3, L2=0.3)
    #         model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy',
    #                       metrics=['accuracy'])  # , lamb = 0.5, sigma = 5)
    #         model.set_hyper(lamb=0.3, sigma=sigma/100)
    #         model.fit(train_dataset, epochs=30)
    #
    #         outputs = model.predict(all_X)
    #         [pred_category, pred_grade] = outputs
    #         results = model.evaluate(test_dataset)
    #
    #
    #         absolute_acc = custom_accuracy([all_y_category,all_y_grade], [pred_category, pred_grade]).numpy()
    #         category_acc = category_accuracy(all_y_category, pred_category).numpy()
    #         grade_acc = grade_accuracy(all_y_grade, pred_grade).numpy()
    #         print(f'Absolute Accuracy: {absolute_acc} \n Category Accuracy: {category_acc} \n Grade Accuracy: {grade_acc}')
    #         temp_abs += absolute_acc
    #         temp_cat += category_acc
    #         temp_grade += grade_acc
    #     result_list.append([temp_abs/5, temp_cat/5, temp_grade/5])
    #
    #     if sigma % 20 == 0:
    #         with open('4label_result_sigma_when_lamb-0.3.txt', 'w') as f:
    #             for item in result_list:
    #                 f.write("%s\n" % item)
    #
    # print(result_list)
    #
    # with open('4label_result_sigma_when_lamb-0.3.txt', 'w') as f:
    #     for item in result_list:
    #         f.write("%s\n" % item)
    #
    # print(result_list)
