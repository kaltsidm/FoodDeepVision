from model import create_model
from data_preperation import class_names

input_shape = (224, 224, 3)
num_classes = len(class_names)

model = create_model(input_shape, num_classes)

model.compile(
    loss = losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ["accuracy"]
)

tf.get_logger().setLevel("ERROR")

# history1: 101 food classes feature extract
history1 = model.fit(train_data,
                    epochs = 3,
                    steps_per_epoch = len(train_data),
                    validation_data = test_data,
                    validation_steps = int(0.15 * len(test_data)),
                    callbacks=[create_tensorboard_callback("training_logs", "efficientnetb0_101_classes_all_data_feature_extract"),
                    model_checkpoint])

