"""
almost copypast from original implementation 
"""
import os
import glob
from absl import logging
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import ml_collections


def get_config():
    """Returns config values other than model parameters."""

    config = ml_collections.ConfigDict()

    # Where to search for pretrained ViT models.
    # Can be downloaded from gs://vit_models/imagenet21k
    config.pretrained_dir = "."
    # Which dataset to finetune on. This can be the name of a tfds dataset
    # (see https://www.tensorflow.org/datasets/catalog/overview), or the path to
    # a directory with the following structure ($filename can be arbitrary):
    # "{train,test}/$class_name/$filename.jpg"
    config.dataset = ""
    # Path to manually downloaded dataset
    config.tfds_manual_dir = None
    # Path to tensorflow_datasets directory
    config.tfds_data_dir = None
    # Number of steps; determined by hyper module if not specified.
    config.total_steps = None

    # Resizes global gradients.
    config.grad_norm_clip = 1.0
    # Datatype to use for momentum state ("bfloat16" or "float32").
    config.optim_dtype = "bfloat16"
    # Accumulate gradients over multiple steps to save on memory.
    config.accum_steps = 8

    # Batch size for training.
    config.batch = 512
    # Batch size for evaluation.
    config.batch_eval = 512
    # Shuffle buffer size.
    config.shuffle_buffer = 50_000
    # Run prediction on validation set every so many steps
    config.eval_every = 100
    # Log progress every so many steps.
    config.progress_every = 10
    # How often to write checkpoints. Specifying 0 disables checkpointing.
    config.checkpoint_every = 1_000

    # Number of batches to prefetch to device.
    config.prefetch = 2

    # Base learning-rate for fine-tuning.
    config.base_lr = 0.03
    # How to decay the learning rate ("cosine" or "linear").
    config.decay_type = "cosine"
    # How to decay the learning rate.
    config.warmup_steps = 500

    # Alternatives : inference_time.
    config.trainer = "train"

    # Will be set from ./models.py
    config.model = None
    # Only used in ./augreg.py configs
    config.model_or_filename = None
    # Must be set via `with_dataset()`
    config.dataset = None
    config.pp = None

    return config.lock()

# We leave out a subset of training for validation purposes (if needed).
DATASET_PRESETS = {
    'cifar10': ml_collections.ConfigDict(
        {'total_steps': 10_000,
         'pp': ml_collections.ConfigDict(
             {'train': 'train[:98%]',
              'test': 'test',
              'crop': 224})
         }),
    'cifar100': ml_collections.ConfigDict(
        {'total_steps': 10_000,
         'pp': ml_collections.ConfigDict(
             {'train': 'train[:98%]',
              'test': 'test',
              'crop': 224})
         }),
    'imagenet2012': ml_collections.ConfigDict(
        {'total_steps': 20_000,
         'pp': ml_collections.ConfigDict(
             {'train': 'train[:99%]',
              'test': 'validation',
              'crop': 224})
         }),
}


def with_dataset(
    config: ml_collections.ConfigDict, dataset: str
) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict(config.to_dict())
    config.dataset = dataset
    config.update(DATASET_PRESETS[dataset])
    return config


def get_tfds_info(dataset, split):
    """Returns information about tfds dataset -- see `get_dataset_info()`."""
    data_builder = tfds.builder(dataset)
    return dict(
        num_examples=data_builder.info.splits[split].num_examples,
        num_classes=data_builder.info.features["label"].num_classes,
        int2str=data_builder.info.features["label"].int2str,
        examples_glob=None,
    )


def get_directory_info(directory):
    """Returns information about directory dataset -- see `get_dataset_info()`."""
    examples_glob = f"{directory}/*/*.jpg"
    paths = glob.glob(examples_glob)
    get_classname = lambda path: path.split("/")[-2]
    class_names = sorted(set(map(get_classname, paths)))
    return dict(
        num_examples=len(paths),
        num_classes=len(class_names),
        int2str=lambda id_: class_names[id_],
        examples_glob=examples_glob,
    )


def get_dataset_info(dataset, split):
    """Returns information about a dataset.

    Args:
      dataset: Name of tfds dataset or directory -- see `./configs/common.py`
      split: Which split to return data for (e.g. "test", or "train"; tfds also
        supports splits like "test[:90%]").

    Returns:
      A dictionary with the following keys:
      - num_examples: Number of examples in dataset/mode.
      - num_classes: Number of classes in dataset.
      - int2str: Function converting class id to class name.
      - examples_glob: Glob to select all files, or None (for tfds dataset).
    """
    directory = os.path.join(dataset, split)
    if os.path.isdir(directory):
        return get_directory_info(directory)
    return get_tfds_info(dataset, split)


def get_data_from_tfds(*, config, mode):
    """Returns dataset as read from tfds dataset `config.dataset`."""

    data_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)

    data_builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=config.tfds_manual_dir)
    )
    data = data_builder.as_dataset(
        split=config.pp[mode],
        # Reduces memory footprint in shuffle buffer.
        decoders={"image": tfds.decode.SkipDecoding()},
        shuffle_files=mode == "train",
    )
    image_decoder = data_builder.info.features["image"].decode_example

    dataset_info = get_tfds_info(config.dataset, config.pp[mode])
    return get_data(
        data=data,
        mode=mode,
        num_classes=dataset_info["num_classes"],
        image_decoder=image_decoder,
        repeats=None if mode == "train" else 1,
        batch_size=config.batch_eval if mode == "test" else config.batch,
        image_size=config.pp["crop"],
        shuffle_buffer=min(dataset_info["num_examples"], config.shuffle_buffer),
    )


def get_data(
    *,
    data,
    mode,
    num_classes,
    image_decoder,
    repeats,
    batch_size,
    image_size,
    shuffle_buffer,
    preprocess=None,
):
    """Returns dataset for training/eval.

    Args:
      data: tf.data.Dataset to read data from.
      mode: Must be "train" or "test".
      num_classes: Number of classes (used for one-hot encoding).
      image_decoder: Applied to `features['image']` after shuffling. Decoding the
        image after shuffling allows for a larger shuffle buffer.
      repeats: How many times the dataset should be repeated. For indefinite
        repeats specify None.
      batch_size: Global batch size. Note that the returned dataset will have
        dimensions [local_devices, batch_size / local_devices, ...].
      image_size: Image size after cropping (for training) / resizing (for
        evaluation).
      shuffle_buffer: Number of elements to preload the shuffle buffer with.
      preprocess: Optional preprocess function. This function will be applied to
        the dataset just after repeat/shuffling, and before the data augmentation
        preprocess step is applied.
    """

    def _pp(data):
        im = image_decoder(data["image"])
        if im.shape[-1] == 1:
            im = tf.repeat(im, 3, axis=-1)
        if mode == "train":
            channels = im.shape[-1]
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(im),
                tf.zeros([0, 0, 4], tf.float32),
                area_range=(0.05, 1.0),
                min_object_covered=0,  # Don't enforce a minimum area.
                use_image_if_no_bounding_boxes=True,
            )
            im = tf.slice(im, begin, size)
            # Unfortunately, the above operation loses the depth-dimension. So we
            # need to restore it the manual way.
            im.set_shape([None, None, channels])
            im = tf.image.resize(im, [image_size, image_size])
            if tf.random.uniform(shape=[]) > 0.5:
                im = tf.image.flip_left_right(im)
        else:
            im = tf.image.resize(im, [image_size, image_size])
        im = (im - 127.5) / 127.5
        label = tf.one_hot(
            data["label"], num_classes
        )  # pylint: disable=no-value-for-parameter
        return {"image": im}, {"label": label}

    data = data.repeat(repeats)
    if mode == "train":
        data = data.shuffle(shuffle_buffer)
    if preprocess is not None:
        data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
    data = data.map(_pp, tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=True)

    '''
    no need to shard since I hane only one device
    '''

    # # Shard data such that it can be distributed accross devices
    # num_devices = 1

    # def _shard(data):
    #     data["image"] = tf.reshape(
    #         data["image"], [num_devices, -1, image_size, image_size, 3]
    #     )
    #     data["label"] = tf.reshape(data["label"], [num_devices, -1, num_classes])
    #     return data

    # if num_devices is not None:
    #     data = data.map(_shard, tf.data.experimental.AUTOTUNE)

    return data.prefetch(1)


if __name__ == "__main__":

    dataset = "cifar10"
    batch_size = 512
    batch_size = 512
    config = with_dataset(get_config(), dataset)
    num_classes = get_dataset_info(dataset, "train")["num_classes"]
    config.batch = batch_size
    config.pp.crop = 224
    ds_train = get_data_from_tfds(config=config, mode='train')
    print("ds_train:",ds_train)

