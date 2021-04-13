import os
import sys
import argparse
from tools.logger import Logger
#from tools.loader_toy_am_box import get_loaders
#from tools.loader_toy_2nd_5_am import get_loaders
from tools.loader_am_2 import get_loaders
from tools.configuration import Configuration
from tools.model_factory import Model



# Train the network
def process(cf):
    # Enable log file
    sys.stdout = Logger(cf.log_file)
    print (' --------- Init experiment: ' + cf.exp_name + ' ---------')

    # Create the data generators
    print ('\n > Creating data generators...')
    train_gen, val_gen, test_gen = get_loaders(cf)

    # Build model
    print ('\n > Building model...')
    model = Model(cf)

    if cf.train_model:
        print()
        # Train the model
        model.train(train_gen, val_gen)      

    if cf.test_model:
        print()
        # Compute validation metrics
        if cf.test_set:
            model.test2(test_gen)
            #model.test_semi_bsl(test_gen)
        else:
            model.test2(val_gen)
        # Compute test metrics
        #model.test(test_gen)

    if cf.pred_model:
        #train_gen, val_gen, _= get_loaders(cf, phase='pred')
        if cf.test_set:
            model.predict(test_gen)
        else:
            model.predict(val_gen)
        #model.predict(train_gen)
        #model.predict_patching(val_gen)


    # Finish
    print (' --------- Finish experiment: ' + cf.exp_name + ' ---------')



def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Creating and training a model to segment Head and Neck tumors')
    parser.add_argument('-c', '--config_path', type=str,
                        default='config.py', help='Configuration file')
    parser.add_argument('-e', '--exp_name', type=str,
                        default=None, help='Name of the experiment')
    parser.add_argument('-l', '--local_path', type=str,
                        default='../', help='Path to local data folder')    
    arguments = parser.parse_args()
    
    # Define the user paths
    
    local_path = arguments.local_path
    dataset_path = os.path.join(local_path, 'F:/data_3d/')#'F:/cropped_for_seg/fov_pre_am_7/')#
    experiments_path = os.path.join(local_path, 'Exp_paper/segmentation/')
    
    # Load configuration files
    configuration = Configuration(arguments.config_path, 
                                  dataset_path,
                                  experiments_path, arguments.exp_name)
    cf = configuration.load()
    
    # Train /test/predict with the network, depending on the configuration
    process(cf)



# Entry point of the script
if __name__ == "__main__":
    main()

#import os
#import sys
#import argparse
#from tools.logger import Logger
##from tools.loader_toy_am_box import get_loaders
#from tools.loader_toy_2nd_5_am import get_loaders
#from tools.configuration import Configuration
#from tools.model_factory import Model
#parser = argparse.ArgumentParser(description='Creating and training a model to segment Head and Neck tumors')
#parser.add_argument('-c', '--config_path', type=str,
#                    default='config.py', help='Configuration file')
#parser.add_argument('-e', '--exp_name', type=str,
#                    default=None, help='Name of the experiment')
#parser.add_argument('-l', '--local_path', type=str,
#                    default='../', help='Path to local data folder')    
#arguments = parser.parse_args()
#
## Define the user paths
#
#local_path = arguments.local_path
#dataset_path = os.path.join(local_path, 'F:/data_3d/')#'F:/cropped_for_seg/fov_pre_am_7/')#
#experiments_path = os.path.join(local_path, 'Exp_paper/segmentation/')
#
## Load configuration files
#configuration = Configuration(arguments.config_path, 
#                              dataset_path,
#                              experiments_path, arguments.exp_name)
#cf = configuration.load()
#print ('\n > Creating data generators...')
#train_gen, val_gen, test_gen = get_loaders(cf)
#traing = iter(train_gen)
#batch = next(traing)
#
#import matplotlib.pyplot as plt
#idx = 50
#plt.imshow(batch[0][0][0][0][:,:,idx]+100*batch[0][1][0][0][:,:,idx],cmap='gray')