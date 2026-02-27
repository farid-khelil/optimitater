
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')



def build_cae_model( obj,encoding_dim=32, dropout_rate=0.5, learning_rate=0.001):
        model = Sequential()


        # ===== ENCODER PART =====
        model.add(Dense(128, activation='relu', input_shape=(obj.n_features,)))
        model.add(BatchNormalization())

        model.add(Dense(encoding_dim, activation='relu'))
        model.add(BatchNormalization())

        # ===== CLASSIFIER PART =====
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_rate))

        model.add(Dense(obj.n_classes, activation='softmax')) 

        # ===== COMPILE =====
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model