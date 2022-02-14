from __future__ import print_function

try:
    raw_input
except:
    raw_input = input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
import random
import pickle
from ffn_model_2D import FFN
import matplotlib.pyplot as plt

print(os.getcwd())

# Commands
# Task parameters
FLAGS = flags.FLAGS
flags.DEFINE_string('datasource', 'spindle', 'Name of datasource to be used, default is spindle')
flags.DEFINE_integer('shot', 100,'Number of training samples per class per task eg. 1-shot refers to 1 training sample per class')
flags.DEFINE_integer('shot_test', 200, 'Number of test samples per class per task')
flags.DEFINE_integer('few_shot', 100, 'Number of shots for testing')
flags.DEFINE_integer('seed', 100, 'Set seed')
flags.DEFINE_float('l1', 1e-5, 'Weights Penalty')
flags.DEFINE_float('l2', 1e-4, 'Weights Penalty')
# Training parameters
flags.DEFINE_integer('epochs', 2, 'Number of metatraining iterations')
flags.DEFINE_integer('batch_size', 1, 'Batchsize for metatraining')
flags.DEFINE_float('lr', 5e-4, 'Meta learning rate')
flags.DEFINE_string('savepath', 'saved_model/', 'Path to save or load models')
flags.DEFINE_string('gpu', '0', 'id of the gpu to use in the local machine')
flags.DEFINE_float('wd', 1e-6, 'weight decay')

## Defines which operation within the model to perform ##
## Only one should be set to True at a time ##
flags.DEFINE_bool('Training', True, '--')
flags.DEFINE_bool('Few_Shot_Training', False, '--')
flags.DEFINE_bool('Testing', False, '--')

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
PRINT_INTERVAL = 1000
TEST_PRINT_INTERVAL = 4000

#Set the number of tasks based on the number of tasks in the training dataset being used, should be either 1160, 2600 or 5200
if FLAGS.datasource == 'spindle':
    num_tasks = 2600
    num_tasks_few_shot = 1
    num_tasks_test = 2600


#Define the main function
def main():
    
    #specify random number starting point, seed set to value of 100
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    
    #Call function to generate the model based on the number training samples,
    #testing samples, L1 and L2 regularization
    model = FFN('model', num_train_samples=FLAGS.shot, num_test_samples=FLAGS.shot_test,
                l1_penalty=FLAGS.l1, l2_penalty=FLAGS.l2)

    #Define Batch
    Batch = tf.Variable(0, dtype=tf.float32)
    
    #Define learning_rate, returns scalar tensor of the same shape as learning_rate input
    learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.lr, global_step=Batch,
                                decay_steps=1e5, decay_rate=0.5, staircase=True)
    
    #Define adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    
    reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name])
    
    #Define loss of model for use in training
    loss = model.loss + FLAGS.wd * reg_term
    
    #Train the optimizer from above with the loss from above and the given batch size
    train_op = optimizer.minimize(loss, global_step=Batch)
    
    config = tf.ConfigProto()
    
    #Dynamically grow memory on GPU
    config.gpu_options.allow_growth = True
    
    #Sets default tensorflow session
    sess = tf.InteractiveSession(config=config)
    
    #Initializes global variables for training
    tf.global_variables_initializer().run()
    
    #Allows the model to save variables
    saver = tf.train.Saver()
    
    #Specify model directory
    directory = os. getcwd()
    
    ## Name of the model to be trained, tested ##
    config_str = 'Model_1'
    
    #config_str += FLAGS.datasource + '_' + str(FLAGS.shot) + 'shot_' + str(FLAGS.batch_size) + 'batch_'+ \
#'l1' + str(FLAGS.l1) + '_l2' + str(FLAGS.l2) + '_lr' + str(FLAGS.lr) + '_wd' + str(FLAGS.wd)

    
    #Creates a save path based on the previously named model
    save_path = FLAGS.savepath + config_str + '/'
    
    ## Import the data loading script ## 
    import reg_data_generator

    #Import data training and testing data from reg_data_generator function ##
    if FLAGS.datasource == 'spindle':
        dataset = reg_data_generator.Spindle_Train_Generator(FLAGS.shot, FLAGS.shot_test, train = True, bmaml = True)
        dataset_test = reg_data_generator.Spindle_Train_Generator(FLAGS.shot, FLAGS.shot_test, train = False, bmaml = True)
    
    
    #tf.summary.scalar('loss_mse', model.plain_loss)
    #tf.summary.scalar('loss', model.loss)
    #tf.summary.scalar('reg', model.penalty_loss)
    #tf.summary.scalar('learning rate', learning_rate)

    #mergerd_sum = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(save_path + 'train_writer', sess.graph)
    

    itr = 0
    #Specifies the number of iterations per epoch based on samples and batch size
    num_iters_per_epoch = int(num_tasks / FLAGS.batch_size)
    print("iterations per epoch", num_iters_per_epoch)
    
  #_____________________________________
  #____________Training Loop____________
  #_____________________________________
    
    if FLAGS.Training == True:
        
        #Loop for the number of epochs
        for e_idx in range(FLAGS.epochs):
         
            # for each batch tasks
            #create a random rearranged array from input
            perm = np.random.permutation(num_tasks)

            #Loop for the number of iterations
            for b_idx in range(num_iters_per_epoch):
                
                # count iter
                index = perm[FLAGS.batch_size * b_idx:FLAGS.batch_size * (b_idx + 1)]

                
                #Specify the batch based on the loaded dataset
                if FLAGS.datasource == 'spindle':
                    train_x, train_y, valid_x, valid_y = dataset.generate_batch(index)
                    
                #Set data dictionary with given batch data    
                feed_dict = {model.train_inputs: train_x, model.train_labels: train_y, model.test_inputs: valid_x, model.test_labels: valid_y}
                
                #Run model evaluation and training
                sess.run(train_op, feed_dict)
                #outs = sess.run([model.loss, train_op, mergerd_sum, model.plain_loss, model.penalty_loss, Batch], feed_dict)
                
                #Update model summary after training iteration
                #train_writer.add_summary(outs[2], outs[5])
                itr += 1
                print(itr)
          
        #Checkpoint, save model after each epoch
        saver.save(sess, directory + '/tmp/' + config_str + ".ckpt")
    #__________________________________________
    #__________Few-Shot Training loop__________ (performed as a separate run of the code)
    #__________________________________________
    
    elif FLAGS.Few_Shot_Training == True:
        
        print('Few-Shot Training')
        
        #Load in the testing data
        dataset_test = reg_data_generator.Spindle_Test_Generator(FLAGS.shot, FLAGS.shot_test, train = True, bmaml = True)

        
        #initialize the saver
        saver = tf.train.Saver()
        
        #Fine Tune epochs (only 1 iteration per epoch)
        FLAGS.epochs = 25
        
        #Fine Tune batch size
        FLAGS.batch_size = 1
        #Number of iterations for few shot 
        num_few_shot_iterations = 10

        # Number of tasks of testing dataset
        num_tasks_few_shot = 1160
        
        #Initialize the tensorflow session
        with tf.Session() as sess:    
            
            #Restore the previously trained model
            saver.restore(sess, directory + '/tmp/' + config_str + ".ckpt")
            print('Model Restored')

            #Initialize the penalty loss and MSE loss lists
            penalty_loss_list = []
            val_mse_list = []
            itr = 0
            
            # for each batch tasks
            # create a random rearranged array from input
            perm_test = np.random.permutation(num_tasks_few_shot)
            print(perm_test)
            
            epoch = 0
            
            #Loop for the number of epochs
            for e_idx in range(FLAGS.epochs):
                
                #Loop for the number of iterations
                for iii in range(num_few_shot_iterations):
                    
                    #Loop through the tasks
                    #for iii in range(4):
                
                    # count iter
                    index = perm_test[FLAGS.batch_size * iii:FLAGS.batch_size * (iii + 1)]

                    ## Load in data for few-shot training ##
                    if FLAGS.datasource == 'spindle':
                        train_x, train_y, valid_x, valid_y = dataset_test.generate_batch(index)

                    #Set data dictionary with given batch data   
                    feed_dict = {model.train_inputs: train_x, model.train_labels: train_y, model.test_inputs: valid_x, model.test_labels: valid_y}

                    #Run model evaluation and training
                    outs = sess.run(train_op, feed_dict)

                    #Update model summary after training iteration
                    #train_writer.add_summary(outs[2], outs[5])
                    #itr = outs[5] + 1
                    #print('iteration',itr)

                epoch += 1
                print('Epoch', epoch)

            #Checkpoint, save model after each epoch
            saver.save(sess, directory + '/tmp/' + config_str + '_Few_Shot' + ".ckpt")
            print('Model :', save_path)
            print("Model Updated & Saved")
            
            with open('Permutation' +'.pkl','wb') as f:
                pickle.dump(perm_test, f)
      
    #_______________________________     
    #________Testing loop___________ (performed as a separate run of the code)
    #_______________________________

    elif FLAGS.Testing == True:
        
        #Load in the testing data
        dataset_test = reg_data_generator.Spindle_Test_Generator(FLAGS.shot, FLAGS.shot_test, train = False, bmaml = True)
        
        #initialize the saver
        saver = tf.train.Saver()
        
        
        #Initialize the tensorflow session
        with tf.Session() as sess:    
            
            #Restore the previously trained model
            saver.restore(sess, directory + '/tmp/' + config_str +  ".ckpt")

            #Restore the previously few-shot trained model
            #saver.restore(sess, directory + '/tmp/' + config_str + '_Few_Shot' +  ".ckpt")
            print('Model Restored')
            
            FLAGS.batch_size = 1
            #Initialize the penalty loss and MSE loss lists
            loss_amplitude_list = []
            loss_phase_list = []
            
            penalty_loss_amplitude_list = []
            penalty_loss_phase_list = []
            
            plain_loss_amplitude_list =[]
            plain_loss_phase_list = []
            
            itr = 0
            model_output = []
            data_label = []
            
            # for each batch tasks
            #create a random rearranged array from input
            perm_test = np.random.permutation(num_tasks_test)
            
            #Loop through all of the testing data
            for iii in range(int(num_tasks_test / FLAGS.batch_size)):
                
                # count iter
                #index = perm_test[FLAGS.batch_size * iii:FLAGS.batch_size * (iii + 1)]
                
                #Specify the batch based on the loaded dataset
                if FLAGS.datasource == 'mnist':
                    train_x, train_y, valid_x, valid_y = dataset_test.generate_batch(iii)
                elif FLAGS.datasource == 'sinusoid':
                    train_x, train_y, valid_x, valid_y = dataset_test.generate_batch(iii)
                elif FLAGS.datasource == 'spindle':
                    train_x, train_y, valid_x, valid_y = dataset_test.generate_batch([iii])
                    
                #Set data dictionary with given batch data   
                feed_dict = {model.train_inputs: train_x, model.train_labels: train_y, model.test_inputs: valid_x, model.test_labels: valid_y}
                
                #Run model evaluation and testing
                outs = sess.run([model.loss_amplitude, model.loss_phase, model.plain_loss_amplitude, model.plain_loss_phase,  model.p1_list, model.p2_list, model.label_list1, model.label_list2], feed_dict)
                
                #Append losses to the appropriate lists
                loss_amplitude_list.append(outs[0])
                loss_phase_list.append(outs[1])
                     
                plain_loss_amplitude_list.append(outs[2])
                plain_loss_phase_list.append(outs[3])
                
                #penalty_loss_amplitude_list.append(outs[4])
                #penalty_loss_phase_list.append(outs[5])
                 
                itr += 1
                print(itr)
                
                a = outs[4]
                a = np.array(a)
                a = a.reshape((100,1))
                
                b = outs[5]
                b = np.array(b)
                b = b.reshape((100,1))
                
                c = outs[6]
                c = np.array(c)
                c = c.reshape((100,1))
                
                d = outs[7]
                d = np.array(d)
                d = d.reshape((100,1))
                
                e = np.concatenate((a,b), axis = 1)
                
                f = np.concatenate((c,d),axis = 1)
                
                model_output.append(e)
                data_label.append(f)
                
            #Define losses and print them
            loss_amplitude = np.mean(np.sqrt(loss_amplitude_list))
            loss_phase = np.mean(np.sqrt(loss_phase_list))
            
            #penalty_loss_amplitude = np.mean(np.sqrt(penalty_loss_amplitude_list))
            #penalty_loss_phase = np.mean(np.sqrt(penalty_loss_phase_list))
            
            plain_loss_amplitude = np.mean(np.sqrt(plain_loss_amplitude_list))
            plain_loss_phase = np.mean(np.sqrt(plain_loss_phase_list))
            
            print('Model :', save_path)
            
            print('Amplitude Losses:',' MRMSE: ', plain_loss_amplitude)
   
            print('Phase Losses:',' MRMSE: ', plain_loss_phase)
    
            print('model_output',np.shape(model_output))
            print('data_label',np.shape(data_label))

            comparison_data = {}
            comparison_data = {'Prediction': model_output, 'Ground_Truth': data_label}
            
            ## Export and save prediction results in a pkl file ##
            with open(config_str+ '_Prediction' + '.pkl','wb') as f:
                pickle.dump(comparison_data, f)

#___________________________________________
#________Run the main function above________
#___________________________________________
if __name__ == '__main__':
    main()
