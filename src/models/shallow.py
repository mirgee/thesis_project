import keras as k
import numpy as np
import tensorflow as tf

def square(x):
    import tensorflow as tf
    return tf.math.square(x)

def safe_log(x):
    import tensorflow as tf
    return tf.math.log(tf.clip_by_value(x, 1e-6, np.inf))

def shal_model(
    in_chans = 19,
    input_time_length = 256,
    n_classes = 2,
    n_filters_time=40,
    n_filters_spat=40,
    filter_time_length=25,
    pool_time_length=75,
    pool_time_stride=15,
    final_conv_length='auto', # 11,
    batch_norm=True,
    batch_norm_alpha=0.1,
    conv_nonlin=square,
    pool_mode='mean',
    pool_nonlin=safe_log,
    drop_prob=0.5,
):
    pool_class = dict(max=k.layers.MaxPooling2D, mean=k.layers.AveragePooling2D)[pool_mode]
    inp = k.layers.Input((in_chans, input_time_length, 1), name='input')
    
    layer = k.layers.Conv2D(
        filters=n_filters_time,
        kernel_size=(1, filter_time_length),
        use_bias=not batch_norm,
        kernel_initializer=k.initializers.glorot_uniform(),
        bias_initializer='zeros',
        data_format='channels_last',
        input_shape=(in_chans, input_time_length, 1),
        name='conv_time'
    )(inp)
    
    layer = k.layers.Conv2D(
        filters=n_filters_spat,
        kernel_size=(in_chans, 1),
        use_bias=not batch_norm,
        kernel_initializer=k.initializers.glorot_uniform(),
        bias_initializer='zeros',
        name='conv_spat'
    )(layer)
    
    # Affine in orig - what it means? TODO: Apply here
    # Also, in orig some initialization
    if batch_norm:
        layer = k.layers.BatchNormalization(
            axis=-1,
            momentum=batch_norm_alpha,
            epsilon=1e-5,
            name='batch_norm',
            # There are other default params which may be diff. from orig.
        )(layer)
    
    layer = k.layers.Lambda(conv_nonlin, name='conv_nonlin')(layer)
    
    layer = pool_class(
        pool_size=(1, pool_time_length),
        strides=(1, pool_time_stride),
        name='pool'
    )(layer)
    
    layer = k.layers.Lambda(pool_nonlin, name='pool_nonlin')(layer)
    
    layer = k.layers.Dropout(drop_prob, name='drop')(layer)
    
    # TODO: Figure out final convolution length
    if final_conv_length == 'auto':
        final_conv_length = int(layer.shape[2])
    
    layer = k.layers.Conv2D(
        filters=n_classes,
        kernel_size=(1, final_conv_length),
        use_bias=True,
        activation='softmax',
        kernel_initializer=k.initializers.glorot_uniform(),
        bias_initializer='zeros',
        name='conv_classifier'
    )(layer)
    
    layer = k.layers.Lambda(lambda x: x[:,0,0,:], name='squeeze')(layer)
    
    model = k.models.Model(inp, layer)
    # model.compile(
    #     optimizer=k.optimizers.SGD(lr=0.01, momentum=0.99, decay=1e-5, nesterov=True),
    #     loss=k.losses.binary_crossentropy,
    #     metrics=['accuracy'],
    #     loss_weights=None,
    #     sample_weight_mode=None,
    #     weighted_metrics=None, 
    #     target_tensors=None
    # )
    
    return model


if __name__ == '__main__':
    print(shal_model().summary())
