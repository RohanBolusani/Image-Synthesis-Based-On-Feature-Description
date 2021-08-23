import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os
import imageio
import warnings
from skimage import img_as_ubyte
import time

def main():
	ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
	model_options = {
		'z_dim' : 100,
		't_dim' : 256,
		'batch_size' : 5,
		'image_size' : 64,
		'gf_dim' : 64,
		'df_dim' : 64,
		'gfc_dim' : 1024,
		'caption_vector_length' : 2400
	}
	
	t1 = time.time()
	gan = model.GAN(model_options)
	_, _, _, _, _ = gan.build_model()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, 'Data/Models/latest_model_flowers_temp.ckpt')

	input_tensors, outputs = gan.build_generator()

	h = h5py.File( 'Data/sample_caption_vectors.hdf5' )
	caption_vectors = np.array(h['vectors'])
	caption_image_dic = {}
	for cn, caption_vector in enumerate(caption_vectors):

		caption_images = []
		z_noise = np.random.uniform(-1, 1, [model_options["batch_size"], model_options['z_dim']])
		caption = [ caption_vector[0:model_options['caption_vector_length']] ] * model_options["batch_size"]
		
		[ gen_image ] = sess.run( [ outputs['generator'] ], 
			feed_dict = {
				input_tensors['t_real_caption'] : caption,
				input_tensors['t_z'] : z_noise,
			} )
		
		caption_images = [gen_image[i,:,:,:] for i in range(0, model_options["batch_size"])]
		t2 = time.time()
		caption_image_dic[ cn ] = caption_images
		print("Generated {} image in time ".format(ordinal(cn+1)), "{0:.3f} s".format((t2-t1)) )

	for f in os.listdir( join('Data', 'val_samples')):
		if os.path.isfile(f):
			os.unlink(join('Data', 'val_samples/' + f))

	for cn in range(0, len(caption_vectors)):
		caption_images = []
		for i, im in enumerate( caption_image_dic[ cn ] ):
			caption_images.append( im )
			caption_images.append( np.zeros((64, 5, 3)) )
		combined_image = np.concatenate( caption_images[0:-1], axis = 1 )
		imageio.imwrite(join('Data', 'val_samples/combined_image_{}.jpg'.format(cn)) , img_as_ubyte(combined_image))
	t3 = time.time()
	print("Program executed in {0:.3f}s".format((t3-t1)))



if __name__ == '__main__':
	main()
