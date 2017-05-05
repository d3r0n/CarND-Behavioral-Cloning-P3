from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class Grayscale(Layer):

    def __init__(self, **kwargs):
        super(Grayscale, self).__init__(**kwargs)

    def build(self, input_shape):
        gray = np.array([0.299,0.587,0.114]).reshape(3,1)
        v_gray = K.variable(value=gray, dtype= K.floatx(), name="grayscale")
        self.trainable = False
        self.non_trainable_weights = [v_gray]
        # self.kernel = v_gray
        super(Grayscale, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.non_trainable_weights[0])

    def compute_output_shape(self, input_shape):
        return (input_shape[0:-1])+ (1,)
