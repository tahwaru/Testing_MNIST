# Minimal example for training a network on MNIST
# inspired by https://www.tensorflow.org/datasets/keras_example
import os
import json
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds

tf.random.set_seed(42)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimal example for training a network on MNIST')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='epochs (default: 3)')
    parser.add_argument('-b', '--batchsize', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('-l', '--layers', type=int, default=1, help='hidden layers (default: 1)')
    parser.add_argument('-u', '--units', type=int, default=128, help='hidden units (default: 128)')
    parser.add_argument('-o', '--outfile', default=None, help='write results to file (default: None)')
    parser.add_argument('--save_weights', default=None, help='Path for saving weights (default: None)')
    parser.add_argument('--load_weights', default=None, help='Path for loading weights (default: None)')
    args = parser.parse_args()

    # setup train and test data
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(args.batchsize)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(args.batchsize)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # create model
    hidden = []
    for i in range(args.layers):
        hidden.append(tf.keras.layers.Dense(args.units, activation='relu'))

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28, 1))] +
                                       hidden +
                                       [tf.keras.layers.Dense(10)])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    if args.load_weights:
        assert os.path.exists(args.load_weights)
        model.load_weights(args.load_weights)

    model.summary()

    # train model
    model.fit(
        ds_train,
        epochs=args.epochs,
        validation_data=ds_test,
    )

    # eval model and save results
    results = model.evaluate(ds_test)
    results_dict = {'loss': results[0], 'accuracy': results[1]}

    if args.save_weights:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_weights)), exist_ok=True)
        model.save_weights(args.save_weights)

    if args.outfile:
        os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
        with open(args.outfile, 'w') as stream:
            json.dump(results_dict, stream, indent=2)
    else:
        print(results_dict)
