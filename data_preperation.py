from config import *
datasets_lists = tfds.list_builders()
# 12 minutes
(train_data, test_data) , df_info = tfds.load(name = "food101",
                                              split = ["train", "validation"],
                                              shuffle_files = True,
                                              as_supervised = True,
                                              with_info = True)


class_names = df_info.features["label"].names


# output info about one of our training sample
for image, label in train_data.take(1):
  print(f"""
  Image shape: {image.shape},
  Image datatype: {image.dtype},
  Target class from Food101 (tensor forms): {label}
  Class name (str form): {class_names[label.numpy()]}
  """)

plt.imshow(image)
plt.title(f"Class: {class_names[label.numpy()]}")
plt.axis(False);

def preprocess_img(image, label, image_shape = 224):
  image = tf.image.resize(image, [image_shape, image_shape])
  #image = image/255. # not necessary as EfficientNetXB model do it internally
  return tf.cast(image, tf.float32), label

preprocessed_img = preprocess_img(image, label)[0]
print(f"Image before preprocessing: {image.shape}, {image.dtype}\n"
      f"Image after preprocessing: {preprocessed_img.shape}, {preprocessed_img.dtype}")
print(f"Image before process {image[:2]}\n and he image after preprocess {preprocessed_img[:2]}")

# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map prepreprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

CHECKPOINT_PATH = "model_checkpoints/cp.weights.h5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH,
                                                      monitor = "val_acc",
                                                      save_best_only = True,
                                                      save_weights_only = True,
                                                      verbose = 0) #dot not print anything



mixed_precision.set_global_policy("mixed_float16")

