import tensorflow as tf
from tensorflow.contrib import slim
import os

from nets import inception
from tensorflow.python.framework.graph_util import convert_variables_to_constants
#from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

def freeze_model(name):
    #checkpoints_dir = '/data/huang/behaviour/data/tfmodel'
    OUTPUT_PB_FILENAME = 'inception_resnet_v2_behaviour_371_9_25_{}k.pb'.format(name[0:3])
    NUM_CLASSES = 371  
    tensorName_v4='InceptionResnetV2/Logits/Predictions'
    
    
    
    with tf.Graph().as_default():
        
        image_placeholder = tf.placeholder(tf.float32, shape=(1,512,512,3), name='input_image')
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            _, probabilities = inception.inception_resnet_v2(image_placeholder,
                                                             num_classes = NUM_CLASSES,
                                                             is_training=False)
    
        #model_path = tf.train.latest_checkpoint(checkpoints_dir)
        model_path = '/data/huang/behaviour/data/tfmodel/model_backup/model.ckpt-{}'.format(name)
    
        init_fn = slim.assign_from_checkpoint_fn(
            model_path,
            slim.get_model_variables())
    
        with tf.Session() as sess:
            # Now call the initialization function within the session
            init_fn(sess)
            constant_graph = convert_variables_to_constants(sess, sess.graph_def, ["input_image",tensorName_v4])
            tf.train.write_graph(constant_graph, '.', OUTPUT_PB_FILENAME, as_text=False)
path = '/data/huang/behaviour/data/tfmodel/model_backup'
model_number = []
model_number_set = set()
for directory, folders, files in os.walk(path):
    for file in files:
        model_number_set.add(file.split('.')[1].split('-')[1])
    break
model_number.extend(model_number_set)
model_number.sort()

for index,item in enumerate(model_number):
    freeze_model(item)
    print(index,item)
    
