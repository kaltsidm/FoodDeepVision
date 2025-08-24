def create_model():
  input_shape = (224, 224, 3)
  base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top = False)
  base_model.trainable = False

  inputs = Input(shape = input_shape, name = "input_layer")
  x = Rescaling(1./255)(inputs) # Remove this redundant rescaling layer
  x = base_model(inputs)
  x = GlobalAveragePooling2D(name = "pooling_layer")(x)
  x = Dense(len(class_names))(x)

  outputs = Activation("softmax", dtype = tf.float32, name = "softmax_float32")(x)
  model = tf.keras.Model(inputs, outputs)

  return model