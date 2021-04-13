import imp
import os
import shutil


class Configuration():
    def __init__(self, config_path,
                       dataset_path,
                       experiments_path, exp_name):

        self.config_path = config_path
        self.exp_name = exp_name
        self.dataset_path = dataset_path
        self.experiments_path = experiments_path

    def load(self):
        config_path = self.config_path
        exp_name = self.exp_name
        experiments_path = self.experiments_path

        # Load configuration file
        cf = imp.load_source('config', config_path)

        # Save extra parameter
        cf.config_path = config_path
        if exp_name !=None:
            cf.exp_name = exp_name
        cf.dataset_path = self.dataset_path
        # Create output folders
        cf.savepath = os.path.join(experiments_path, cf.exp_name)
        cf.log_file = os.path.join(cf.savepath, "logfile.log")
        
        if not os.path.exists(cf.savepath):
            os.makedirs(cf.savepath)

        # Copy config file
        if cf.train_model:
            shutil.copyfile(config_path, os.path.join(cf.savepath, "config.py"))

        # Get training weights file name
        path, _ = os.path.split(cf.weights_file)
        if path == '':
            cf.weights_file = os.path.join(cf.savepath, cf.weights_file)

        self.configuration = cf
        return cf

