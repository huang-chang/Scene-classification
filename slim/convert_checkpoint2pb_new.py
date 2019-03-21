import tensorflow as tf
from tensorflow.contrib import slim

from nets import inception
from tensorflow.python.framework.graph_util import convert_variables_to_constants
#from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

checkpoints_dir = '/data/huang/behaviour/data/tfmodel'
OUTPUT_PB_FILENAME = 'inception_resnet_v2_behaviour_309_5_25_434k_rgb.pb'
NUM_CLASSES = 337 
tensorName_v4='InceptionResnetV2/Logits/Predictions'

image_size = inception.inception_resnet_v2.default_image_size

with tf.Graph().as_default():
    
    image_placeholder = tf.placeholder(tf.uint8, shape=(None,None,3), name='input_image')
    image_rgb = tf.reverse(image_placeholder, [-1])
    image = tf.image.convert_image_dtype(image_rgb, dtype=tf.float32)
    image = tf.expand_dims(image,0)
    image = tf.image.resize_bilinear(image, [image_size, image_size],
                                           align_corners=False) 
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    image = tf.expand_dims(image,0)
    print(image)
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        _, probabilities = inception.inception_resnet_v2(image,
                                                  num_classes=NUM_CLASSES,
                                                  is_training=False)

    #model_path = tf.train.latest_checkpoint(checkpoints_dir)
    model_path = '/data/huang/behaviour/data/tfmodel/tfmodel_434k/model.ckpt-434651'

    # Get the function that initializes the network structure (its variables) with
    # the trained values contained in the checkpoint
    init_fn = slim.assign_from_checkpoint_fn(
        model_path,
        slim.get_model_variables())

    with tf.Session() as sess:
        # Now call the initialization function within the session
        init_fn(sess)

        # Convert variables to constants and make sure the placeholder input_image is included
        # in the graph as well as the other neccesary tensors.
        constant_graph = convert_variables_to_constants(sess, sess.graph_def, ["input_image",tensorName_v4])

        # Define the input and output layer properly
        #optimized_constant_graph = optimize_for_inference(constant_graph, ["eval_image"],
        #                                                  [tensorName_v4],
         #                                                 tf.string.as_datatype_enum)
        # Write the production ready graph to file.
        tf.train.write_graph(constant_graph, '.', OUTPUT_PB_FILENAME, as_text=False)
