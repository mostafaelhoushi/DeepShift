from tensorflow.keras import backend as K
import numpy as np

def get_shift_and_sign(x):
    sign = K.sign(x)
    x_abs = K.abs(x)
    shift = K.round(K.log(x_abs) / np.log(2))

    return shift, sign    

def round_power_of_2(x):
    shift, sign = get_shift_and_sign(x)    
    x_rounded = K.pow(2.0, shift) * sign
    return x_rounded
