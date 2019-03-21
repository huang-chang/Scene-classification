from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import threading
import imghdr
import datetime 
import sys

import tensorflow as tf
from datasets import dataset_utils

# The number of images in the validation set.
_NUM_VALIDATION = 0

# Seed for repeatability.
_RANDOM_SEED = 123456

# The number of shards per dataset split.
_NUM_SHARDS = 8

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    'mydata',
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    '/data/huang/behaviour/data',
    'The directory where the output TFRecords and temporary files are saved.')

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    #Create a single Session to run all image coding calls.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

    #Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string) 
    image = tf.image.decode_png(self._png_data,channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image,format = 'rgb',quality =100)  
      
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self,image_data):
    image = self.decode_jpeg(image_data)
    return image.shape[0], image.shape[1]
    
  def png_to_jpeg(self,image_data):
      return self.sess.run(self._png_to_jpeg,feed_dict={self._png_data,image_data})
      
  def decode_jpeg(self,image_data):
    image = self.sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    
    Args:
      filename:string,path of the image file.
      
    Returns:
      boolean indicating if the image is a PNG.
    """
    #return '.png' in filename
    return 'png' == imghdr.what(filename)
    
def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  flower_root = os.path.join(dataset_dir,'data_photos')
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'mydata_%s_%01d-of-%01d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def sub_convert_dataset(dataset_dir,split_name,shard_id,num_per_shard,filenames,class_names_to_ids):
    """converts the per dataset in a thread"""
    image_reader = ImageReader()
    output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
        per_long = end_ndx - start_ndx
        count = 0
        count_number = 0
        
        for i in range(start_ndx, end_ndx):
            #print('Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
            
            try:
                image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                if _is_png(filenames[i]):
                    image_data =image_reader.png_to_jpeg(image_data)
                    print('png image: {}'.format(filenames[i]))
                    
                height, width = image_reader.read_image_dims(image_data)
            except:
                count_number += 1
                print('{}:image error !!!'.format(count_number))
                continue
        
            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, 'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())
            
            count +=1
            if not count % 1000:
                now_time = datetime.datetime.now()
                print('{}-{}-{} thread {}: Processed {} of {} images in thread batch.'.format(
                       now_time.year,now_time.month,now_time.day, shard_id, count, per_long))
                sys.stdout.flush()
            
def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
  
  coord = tf.train.Coordinator()
  threads = []
  
  for shard_id in range(_NUM_SHARDS):
          
    args = (dataset_dir,split_name,shard_id,num_per_shard,filenames,class_names_to_ids)
    t = threading.Thread(target=sub_convert_dataset, args=args)
    t.start()
    threads.append(t)
  
  #Wait for all the threads to terminate
  coord.join(threads)
  
def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[_NUM_VALIDATION:]
  #validation_filenames = photo_filenames[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir)
  #_convert_dataset('validation', validation_filenames, class_names_to_ids,
   #                dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('Finished converting the mydata dataset!')
  
def main(unused_argv):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if FLAGS.dataset_name == 'mydata':
      run(FLAGS.dataset_dir)
  else:
    raise ValueError(
        'dataset_name [%s] was not recognized.' % FLAGS.dataset_dir)

if __name__ == '__main__':
  tf.app.run()
