import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras_tuner import HyperModel, RandomSearch

data_mineral = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_mineral = data_mineral.flow_from_directory('minet', target_size=(64, 64), batch_size=32, class_mode='categorical', subset='training')
val = data_mineral.flow_from_directory('minet', target_size=(64, 64), batch_size=32, class_mode='categorical', subset='validation')

class mineral_hypermodel(HyperModel):
    
    def build(self, hp):
        tuned_model = Sequential()
        tuned_model.add(Conv2D(filters=hp.Int('conv_1_filters', min_value=32, max_value=128, step=32),
                         kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
                         activation='relu',
                         input_shape=(64, 64, 3)))
        tuned_model.add(MaxPooling2D(pool_size=(2, 2)))
        tuned_model.add(Conv2D(filters=hp.Int('conv_2_filters', min_value=32, max_value=128, step=32),
                         kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
                         activation='relu'))
        tuned_model.add(MaxPooling2D(pool_size=(2, 2)))
        tuned_model.add(Flatten())
        tuned_model.add(Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
                        activation='relu', input_shape=(64, 64, 3)))
        tuned_model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
        tuned_model.add(Dense(len(train_mineral.class_indices), activation='softmax'))
        
        tuned_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return tuned_model

hypr_tuner = RandomSearch(
   mineral_hypermodel(),
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='mineral_cnn_hyper',
    project_name='cnn_tuning'
)

hypr_tuner.search(train_mineral, validation_data=val, epochs=10)
best_cnn_model = hypr_tuner.get_best_models(num_models=1)[0]
hypr_tuner.search(train_mineral, validation_data=val, epochs=20)
best_cnn_model.save('mineral_cnn_tuned.h5')