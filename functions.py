import matplotlib.pyplot as plt
import datetime
import tensorflow as tf

def plot_loss(history):
    """
        That function plots 
        the training loss and the 
        validation loss
    
        The argument of that function is
        the history. THe history is created
        after fitting the model to the training
        data. History contains details about the 
        training process, including metrics
        like loss and accuracy for both
        training and validation datasets.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    plt.plot(epochs, loss, label = "training_loss")
    plt.plot(epochs, val_loss, labels = "val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label = "training_accuracy")
    plt.plot(epochs, val_accuracy, label = "val_accuracay")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def compare_history(original_history, new_history, initial_epochs = 5):
    """
    Comparing the histories of two different
    trained models. The parameters that will be compared
    are the loss value, the accuracy value.
    """

    accuracy = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    validation_accuracy = original_history.history["val_accuracy"]
    validation_loss = original_history.history["val_loss"]

    total_accuracy = accuracy + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]
    total_validation_accuracy = validation_accuracy+new_history.history["val_accuracy"]
    total_validation_loss = validation_loss + new_history.history["val_loss"]

    plt.figure(figsize = (8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_accuracy, labels = "Training Accuracy")
    plt.plot(total_validation_accuracy, labels = "Validation Accuracy")
    plt.plot([initial_epochs-1, initial_epochs-1], 
             plt.ylim(), label = "Start Fine Tuning")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, labels = "Training Loss")
    plt.plot(total_validation_loss, label = "Validation Loss")
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label = "start Fine Tuning")
    plt.legend(loc = "upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()


    
def tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log-files to: {log_dir}")
  return tensorboard_callback

