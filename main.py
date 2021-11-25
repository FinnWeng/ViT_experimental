'''
prefetchDataset shapes: {image: (1, 512, 224, 224, 3), label: (1, 512, 10)}, types: {image: tf.float32, label: tf.float32}

'''

from os import name
import ml_collections
import tensorflow as tf
import tensorflow_addons as tfa
import math

from net.vit import ViT
from dataloader import get_data_from_tfds, get_dataset_info

from SAM import dual_vector

import training_config
import model_config

import math

import cv2
import numpy as np


class Sigmoid_Xent_with_Logit(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        loss = tf.reduce_sum(loss, axis = [1,2,3])
        return tf.reduce_mean(loss)

class With_SAM_Model(tf.keras.Model):
    def __init__(self, inputs, outputs , dual_vector, rho, gradient_clipping, no_weight_decay_on_bn = False, l2_reg = 0):
        super(With_SAM_Model, self).__init__(inputs,outputs)
        self.dual_vector_fn = dual_vector
        self.rho = rho
        self.gradient_clipping = gradient_clipping
        self.no_weight_decay_on_bn = no_weight_decay_on_bn
        self.l2_reg = l2_reg
    
    def weights_decay(self, weight_penalty_params):
        if self.no_weight_decay_on_bn:
            weight_l2 = sum(tf.nest.map_structure(lambda x: tf.reduce_sum(tf.math.square(x)),  [ w for w in weight_penalty_params if len(w.shape) > 1]))
        else:
            weight_l2 = sum(tf.nest.map_structure(lambda x: tf.reduce_sum(tf.math.square(x)), weight_penalty_params))
        
        return weight_l2

    
    def get_sam_gradient(self, grads, x, y):

        grads = dual_vector(grads)

        inner_trainable_vars = tf.nest.map_structure(lambda a: tf.identity(a) , self.trainable_variables)
  
        # import pdb
        # pdb.set_trace()


        _ = tf.nest.map_structure(lambda a, b: a.assign(a + self.rho * b), self.trainable_variables , grads) # model to noised model



        with tf.GradientTape() as noised_tape:
            noised_y_pred = self(x)  # Forward pass
            noised_loss = self.compiled_loss(y, noised_y_pred)
            noised_loss +=  self.l2_reg * 0.5 * self.weights_decay(self.trainable_variables)
        
        noised_vars = self.trainable_variables
        noised_grads = noised_tape.gradient(noised_loss, noised_vars)

        _ = tf.nest.map_structure(lambda a, b: a.assign(b), self.trainable_variables, inner_trainable_vars) # noised model to model


        return noised_grads




    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        result = {m.name: m.result() for m in self.metrics}
        x, y = data


        with tf.GradientTape() as tape:

            
            y_pred = self(x)  # Forward pass
            # Compute the loss valuese
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred)
            # loss += self.l2_reg * 0.5 * self.weights_decay(self.trainable_variables)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        if self.rho > 0:  # SAM loss
            gradients = self.get_sam_gradient(gradients, x, y)
            gradients, gradient_norm = tf.clip_by_global_norm(gradients, clip_norm = self.gradient_clipping)
            result["gradient_norm"] = gradient_norm
        else:
            pass

        


            
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value    

        param_norm = tf.math.sqrt(sum(tf.nest.map_structure(lambda x: tf.reduce_sum(tf.math.square(x)), self.trainable_variables)))
        result["param_norm"] = param_norm
        
        return result




class Warmup_Cos_Decay_Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, cos_initial_learning_rate, warmup_steps, cos_decay_steps, alpha = 0):
        self.cos_initial_learning_rate = cos_initial_learning_rate
        self.warmup_steps = warmup_steps
        self.cos_decay_steps = cos_decay_steps
        self.alpha = alpha

        '''
        there's only one step
        the step of warm up and decay step counts separatly.
        when step > warmup step, switch to cos decay model
        '''
    
    def decayed_learning_rate(self, step):
        step = min(step - self.warmup_steps, self.cos_decay_steps) # here i deal with problem of step that count warm up step.
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * step /  self.cos_decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.cos_initial_learning_rate * decayed

    def __call__(self, step):
        
        if step <= self.warmup_steps:
            lr = self.cos_initial_learning_rate*(step/self.warmup_steps)
        else:
            lr = self.decayed_learning_rate(step)
             
        return lr

if __name__ == "__main__":
    # tf.config.experimental_run_functions_eagerly(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[1], 'GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # initialize dataset
    dataset = "cifar10"
    
    config = training_config.with_dataset(training_config.get_config(), dataset)
    ds_train_info = get_dataset_info(dataset, "train")
    ds_train_num_classes = ds_train_info['num_classes']
    ds_train_num_examples = ds_train_info["num_examples"]
    ds_train = get_data_from_tfds(config=config, mode='train')

    ds_val_info = get_dataset_info(dataset, "test")
    ds_val_num_classes = ds_train_info['num_classes']
    ds_val_num_examples = ds_train_info["num_examples"]
    ds_val = get_data_from_tfds(config=config, mode='test')


    one_train_data = next(ds_train.as_numpy_iterator())[0]
    print("one_train_data.shape:", one_train_data["image"].shape) # vit_model_config 
    print(one_train_data["image"].shape[1:])


    # initialize model
    # vit_model_config = model_config.get_b32_config()
    vit_model_config = model_config.get_b16_config()
    print(vit_model_config )
    
    vit_model = ViT(num_classes=ds_train_num_classes, **vit_model_config)

    

    # this init the model and avoid manipulate weight in graph(if using resnet)
    trial_logit = vit_model(one_train_data["image"], train = True) # (512, 10) 

    # for varb in vit_model.trainable_variables:
    #     if "kernel" or "bias" in varb.name:
    #         print("varb name:", varb.name)
    #         print("varb.shape:", varb.shape)
    
    # vit_varb = vit_model.trainable_variables

    # vit_varb[0] = vit_varb[0]*0
    
    # for new_varb, origin_varib in zip(vit_varb, vit_model.trainable_variables):
    #     origin_varib.assign(new_varb)

    # print("vit_varb[0]:",vit_varb[0])
    # print("vit_model.trainable_variables:",vit_model.trainable_variables[0])

    # build model, expose this to show how to deal with dict as fit() input
    model_input = tf.keras.Input(shape=one_train_data["image"].shape[1:],name="image",dtype=tf.float32)

    logit = vit_model(model_input)
    logit = (logit+1)/2

    
    # logit = sam_model(model_input)

    # prob = tf.keras.layers.Softmax(axis = -1, name = "label")(logit)

    # model = tf.keras.Model(inputs = [model_input],outputs = [logit], name = "ViT_model")

    sam_model_config = model_config.get_sam_config()

    sam_model = With_SAM_Model(inputs ={"image":model_input},outputs = {"recon":logit}, dual_vector = dual_vector, rho = sam_model_config.rho,\
        gradient_clipping = sam_model_config.gradient_clipping, l2_reg = sam_model_config.weight_decay)

    model = sam_model



    '''
    the training config is for fine tune. I use my own config instead for training purpose.
    
    '''
    # my training config:
    steps_per_epoch = ds_train_num_examples//config.batch
    validation_steps = 3
    # log_dir="./tf_log/"
    # log_dir="./tf_log/sam_vit/"
    log_dir="./tf_log/origin_vit/"
    total_steps = 100
    warmup_steps = 1000
    base_lr = 1e-3
    epochs = 1000

    # define callback 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10, update_freq= 10)
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/ViT.ckpt',
        save_weights_only= True,
        verbose=1)

    callback_list = [tensorboard_callback,save_model_callback]

    sigmoid_xent = Sigmoid_Xent_with_Logit()

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-2, decay_steps = 1000, decay_rate = 0.01, staircase=False, name=None)
    # lr_schedule = Cosine_Decay_with_Warm_up(base_lr, total_steps, warmup_steps)

    lr_schedule = Warmup_Cos_Decay_Schedule(base_lr, warmup_steps = warmup_steps, cos_decay_steps = steps_per_epoch*epochs)


    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate = base_lr), 
    #     loss={"label":tf.keras.losses.CategoricalCrossentropy(from_logits=False)},
    #     metrics={'label': 'accuracy'}
    #     )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = base_lr), 
        loss={"recon":sigmoid_xent},
        metrics = {"recon":tf.keras.metrics.BinaryCrossentropy(from_logits = True)}
        )

    # print(model.summary())

    # import pdb
    # pdb.set_trace()

    # import pdb
    # pdb.set_trace()

    model.load_weights('./model/ViT.ckpt')

    hist = model.fit(ds_train,
                epochs=epochs, 
                steps_per_epoch=steps_per_epoch,
                validation_data = ds_val,
                validation_steps=1,callbacks = callback_list).history



    one_val_data = next(ds_val.as_numpy_iterator()) # (256, 224, 224, 3)

    logits = model(one_val_data[0]["image"])["recon"]
    logits = tf.keras.activations.sigmoid(logits).numpy() # (8, 224, 224, 3)
    print("logts:",logits.shape)

    print("logts:",logits.shape)



    test_recon =one_val_data[1]["recon"] # #(8, 224, 224, 3)
    print("test_recon:", test_recon.shape)
    inference_img = np.concatenate([test_recon,logits], axis = 2)  #(10, 224, 224*2, 3)
    inference_img = inference_img.reshape([8*224, 224*2, 3])*255
    # print(inference_img.shape)

    cv2.imwrite("inference_img.png", inference_img)