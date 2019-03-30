import keras as k
import numpy as np
import tensorflow as tf

def deep_model(
        in_chans=19,
        n_classes=2,
        input_time_length=256,
        final_conv_length='auto', # 1,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        filter_length_2=5,
        n_filters_3=100,
        filter_length_3=5,
        n_filters_4=200,
        filter_length_4=5,
        first_nonlin=k.activations.elu,
        first_pool_mode='max',
        first_pool_nonlin=(lambda x: x),
        later_nonlin=k.activations.elu,
        later_pool_mode='max',
        later_pool_nonlin=(lambda x: x),
        drop_prob=0.5,
        double_time_convs=False,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False,
):
    if stride_before_pool:
        conv_stride = pool_time_stride
        pool_stride = 1
    else:
        conv_stride = 1
        pool_stride = pool_time_stride
    # TODO: Behavior of their Avg differs
    pool_class_dict = dict(max=k.layers.MaxPool2D, mean=k.layers.AvgPool2D) # AvgPool2dWithConv)
    first_pool_class = pool_class_dict[first_pool_mode]
    later_pool_class = pool_class_dict[later_pool_mode]

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
    
    layer = k.layers.Activation(first_nonlin, name='conv_nonlin')(layer)
    
    layer = first_pool_class(
        pool_size=(1, pool_time_length),
        strides=(1, pool_time_stride),
        name='pool'
    )(layer)
    
    layer = k.layers.Lambda(first_pool_nonlin, name='pool_nonlin')(layer)
    
    def add_conv_pool_block(layer, n_filters, filter_length, block_nr):
        suff = '_{:d}'.format(block_nr)
        layer = k.layers.Dropout(drop_prob, name='drop'+suff)(layer)

        layer = k.layers.Conv2D(
            filters=n_filters,
            kernel_size=(1, filter_length), # Not sure 
            strides=(1, conv_stride), # Not sure 
            use_bias=not batch_norm,
            kernel_initializer=k.initializers.glorot_uniform(),
            bias_initializer='zeros',
            name='conv'+suff
        )(layer)

        if batch_norm:
            layer = k.layers.BatchNormalization(
                axis=-1,
                momentum=batch_norm_alpha,
                epsilon=1e-5,
                name='batch_norm'+suff,
                # There are other default params which may be diff. from orig.
            )(layer)

        layer = k.layers.Activation(later_nonlin, name='nonlin'+suff)(layer)

        layer = later_pool_class(
            pool_size=(1, pool_time_length),
            strides=(1, pool_stride),
            name='pool'+suff
        )(layer)

        layer = k.layers.Lambda(later_pool_nonlin, name='pool_nonlin'+suff)(layer)

        return layer
    
    layer = add_conv_pool_block(layer, n_filters_2, filter_length_2, 2)
    layer = add_conv_pool_block(layer, n_filters_3, filter_length_3, 3)
    layer = add_conv_pool_block(layer, n_filters_4, filter_length_4, 4)

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

