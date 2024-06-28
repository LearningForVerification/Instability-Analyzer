from InstabilityInspector.Inspector import Inspector
import tensorflow as tf
import tensorflow_datasets as tfds

DATASET = "fashion_mnist"


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32` and flattens to 784."""
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    image = tf.reshape(image, [-1])  # Flatten to 1D vector of length 784
    return image, label

if __name__ == '__main__':
    # generate dataset
    (ds_train, ds_test), ds_info = tfds.load(
        DATASET,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    inspector = Inspector("experiments/model.onnx", "experiments", ds_test)
    dict = inspector.bounds_inspector(100, 0.05, 0.15, False, True)
