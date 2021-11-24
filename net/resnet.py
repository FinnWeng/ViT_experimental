import math
import functools

import tensorflow as tf
import tensorflow_addons as tfa

def weight_standardize(w, axis, eps):
  """Subtracts mean and divides by standard deviation."""
  w = w - tf.math.reduce_mean(w, axis=axis)
  w = w / (tf.math.reduce_std(w, axis=axis) + eps)
  return w

class Std_Conv(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Std_Conv, self).__init__()
        self.weight_standardized = False
        self.conv1 = tf.keras.layers.Conv2D(*args, **kwargs)
    def call(self, input):
        if self.weight_standardized == False:
            self.conv1(input)
            old_w = self.conv1.get_weights()
            new_kernel_w = weight_standardize(old_w[0],axis=[0,1,2],eps = 1e-5) 
            # print(old_w[0])
            # print(new_kernel_w)
            self.conv1.set_weights(old_w)
            self.weight_standardized = True
            output = self.conv1(input)
            # print(self.conv1.get_weights()[0])

        else:
            output = self.conv1(input)
        # print("final kernel w", self.conv1.get_weights()[0])
        return output

class Residual_Unit(tf.keras.Model):
    """Bottleneck ResNet block."""
    def __init__(self, features, strides, name=None):
        super(Residual_Unit, self).__init__(name)

        self.features = features
        self.strides = strides

    
    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

    
    def call(self, x):
        needs_projection = (
            x.shape[-1] != self.features * 4 or self.strides != (1, 1))

        residual = x
        if needs_projection:
            residual = self.get("conv_proj",Std_Conv, self.features*4, (1,1),self.strides, 
                padding = "same",use_bias = False, kernel_initializer = tf.keras.initializers.LecunNormal())(residual)
            residual = self.get('gn_proj', tfa.layers.GroupNormalization, epsilon = 1e-6)(residual)

        y = self.get("conv1",Std_Conv, self.features, (1,1), 
                padding = "same",use_bias = False, kernel_initializer = tf.keras.initializers.LecunNormal())(x)
            
        y = self.get("gn1", tfa.layers.GroupNormalization, epsilon = 1e-6)(y)
        y = tf.nn.relu(y)

        y = self.get("conv2",Std_Conv, self.features, (3,3),self.strides, 
            padding = "same",use_bias = False, kernel_initializer = tf.keras.initializers.LecunNormal())(x)
        
        y = self.get('gn2', tfa.layers.GroupNormalization, epsilon = 1e-6)(y)
        y = tf.nn.relu(y)

        y = self.get("conv3", Std_Conv, self.features*4, (1,1), 
            padding = "same",use_bias = False, kernel_initializer = tf.keras.initializers.LecunNormal())(x)

        y = self.get('gn3', tfa.layers.GroupNormalization, epsilon = 1e-6, gamma_initializer= tf.keras.initializers.Zeros())(y)
        y = tf.nn.relu(residual + y)
        return y


class Res_Net_Stage(tf.keras.Model):
    """A ResNet stage."""

    def __init__(self, block_size, nout, first_stride, name=None):
        super(Res_Net_Stage, self).__init__(name)
        self.block_size = block_size
        self.nout = nout
        self.first_stride = first_stride
    
    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

    
    def call(self, x):
        x = self.get('unit1', Residual_Unit, self.nout, strides=self.first_stride)(x)
        for i in range(1, self.block_size):
            x = self.get(f'unit{i + 1}', Residual_Unit, self.nout, strides=(1, 1))(x)
        return x

class Test_Res_Net_Stage(tf.keras.Model):
    def __init__(self, block_size, nout, first_stride, name=None):
        super(Test_Res_Net_Stage, self).__init__(name)
        self.stage1 = Res_Net_Stage(
            block_size=block_size,
            nout=nout,
            first_stride=first_stride,
            name='block1')
        self.d1 = tf.keras.layers.Dense(1)
    
    def call(self, input):
        x = self.stage1(input)
        x = tf.keras.layers.Flatten()(x)
        x = self.d1(x)
        return x 


    

        
    

if __name__ == "__main__":
    '''
    test std_conv    
    '''
    # conv1 = Std_Conv(1,2,1)
    # a = tf.ones([5,256,256,3])
    # b = conv1(a)
    # b = conv1(a) # check if the kernel modify twice 
    '''
    test res stage
    '''
    x =  tf.ones([1,256,256,64])
    y =  tf.ones([1])
    stage1 = Test_Res_Net_Stage(
                    block_size=5,
                    nout=64,
                    first_stride=(1, 1),
                    name='block1')

    stage1(x)
    
    stage1.compile(optimizer="Adam", loss="mse")
    
    stage1.fit(x,y,1,1)
    print(stage1.summary())
    import pdb
    pdb.set_trace()
    '''
    the use "stage1.get_layer(index = 0).get_layer(index = 0).summary()" to check model. looks fine.
    
    '''

