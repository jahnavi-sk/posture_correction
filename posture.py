import tensorflow as tf
def heatmap_subnetwork(input_shape, num_filters) :
# Define the input layer
    inputs = tf. keras. Input (shape=input_shape)
    # Define the convolutional layers
    x = tf. keras.layers.Conv2D(num_filters, kernel_size=3, padding=' same', activation='relu' ) (inputs)

    x = tf. keras.layers.Conv2D(num_filters, kernel_size=3, padding=' same', activation='relu') (x)

    x = tf. keras. layers. Conv2D (num_filters, kernel_size=3, padding ='same', activation= 'relu')(x)

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding ='same', activation= 'relu')(x)

    # Define the output layer
    outputs = tf.keras.layers.Conv2D(1, kernel_size=1, padding='valid', activation='sigmoid') (x)
    # Define the model
    model = tf. keras.Model (inputs=inputs, outputs=outputs)
    return model






