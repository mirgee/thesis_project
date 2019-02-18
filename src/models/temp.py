import keras as k
import numpy as np
import tensorflow as tf

def temp_model(
        imsize=256,
        n_classes=2,
        input_time_length=256,
        final_conv_length='auto',
        n_filters_1=25,
        filter_length_1=10,
        n_filters_2=50,
        filter_length_2=5,
        n_filters_3=100,
        filter_length_3=5,
        n_filters_4=200,
        filter_length_4=5,
        pool_length=3,
        pool_stride=3,
        first_nonlin=k.activations.relu,
        first_pool_mode='max',
        first_pool_nonlin=(lambda x: x),
        later_nonlin=k.activations.relu,
        later_pool_mode='max',
        later_pool_nonlin=(lambda x: x),
        drop_prob=0.5,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False,
):
    if stride_before_pool:
        conv_stride = pool_stride
        pool_stride = 1
    else:
        conv_stride = 1
        pool_stride = pool_stride
    pool_class_dict = dict(max=k.layers.MaxPool2D, mean=k.layers.AvgPool2D) # AvgPool2dWithConv)
    first_pool_class = pool_class_dict[first_pool_mode]
    later_pool_class = pool_class_dict[later_pool_mode]

    inp = k.layers.Input((imsize, imsize, 1), name='input')
    
    layer = k.layers.Conv2D(
        filters=n_filters_1,
        kernel_size=(filter_length_1, filter_length_1),
        use_bias=not batch_norm,
        kernel_initializer=k.initializers.glorot_uniform(),
        bias_initializer='zeros',
        data_format='channels_last',
        input_shape=(imsize, imsize, 1),
        name='conv_1'
    )(inp)
    
    if batch_norm:
        layer = k.layers.BatchNormalization(
            axis=-1,
            momentum=batch_norm_alpha,
            epsilon=1e-5,
            name='batch_norm_1',
        )(layer)
    
    layer = k.layers.Activation(first_nonlin, name='nonlin_1')(layer)
    
    layer = first_pool_class(
        pool_size=(pool_length, pool_length),
        strides=(pool_stride, pool_stride),
        name='pool_1'
    )(layer)
    
    layer = k.layers.Lambda(first_pool_nonlin, name='pool_nonlin_1')(layer)
    
    def add_conv_pool_block(layer, n_filters, filter_length, block_nr):
        suff = '_{:d}'.format(block_nr)
        layer = k.layers.Dropout(drop_prob, name='drop'+suff)(layer)

        layer = k.layers.Conv2D(
            filters=n_filters,
            kernel_size=(filter_length, filter_length), # Not sure 
            strides=(conv_stride, conv_stride), # Not sure 
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
            pool_size=(pool_length, pool_length),
            strides=(pool_stride, pool_stride),
            name='pool'+suff
        )(layer)

        layer = k.layers.Lambda(later_pool_nonlin, name='pool_nonlin'+suff)(layer)

        return layer
    
    layer = add_conv_pool_block(layer, n_filters_2, filter_length_2, 2)
    layer = add_conv_pool_block(layer, n_filters_3, filter_length_3, 3)
    layer = add_conv_pool_block(layer, n_filters_4, filter_length_4, 4)
    # layer = add_conv_pool_block(layer, n_filters_5, filter_length_5, 5)
    
    # layer = k.layers.Flatten()(layer)
    # layer = k.layers.Dense(32, activation='elu')(layer)
    # 
    # layer = k.layers.Dense(n_classes, activation='softmax')(layer)
    
    if final_conv_length == 'auto':
        final_conv_length = int(layer.shape[2])
        
    layer = k.layers.Conv2D(
        filters=n_classes,
        kernel_size=(final_conv_length, final_conv_length),
        use_bias=True,
        activation= 'softmax',
        kernel_initializer=k.initializers.glorot_uniform(),
        bias_initializer='zeros',
        name='conv_classifier'
    )(layer)
    
    # layer = k.layers.Softmax(axis=-1, name='softmax')(layer)
    
    layer = k.layers.Lambda(lambda x: x[:,0,0,:], name='squeeze')(layer)
    
    model = k.models.Model(inp, layer)
    model.compile(
        optimizer=k.optimizers.SGD(lr=0.001, momentum=0.99, decay=1e-5, nesterov=True),
        loss=k.losses.binary_crossentropy,
        metrics=['accuracy'],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None, 
        target_tensors=None
    )
    
    return model
