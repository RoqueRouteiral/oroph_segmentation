
exp_name                     = 'and_baseline_t1c_t2_good_nrm_2'
# Model
model_name                   = 'unet3d'          # Different networks: Unet, DeepMedic, HighRes3dNet, ScaleNet, Vnet
show_model                   = 0             # Show the architecture layers (Either with summary or just with str)
weights_file                 = 'C:/Users/r.rodriguez.outeiral/Documents/Experiments/debug_toy_all_directions_15_14_debug//snapshot_epoch62.ckpt'              # Training weight file name (for finetuning)
weights_test_file            = 'C:/Users/r.rodriguez.outeiral/Documents/Exp_paper/segmentation/new_shifts_8_35_gs/snapshot_epoch44.ckpt'
#E:/Exps_paper/Exp_paper/segmentation/and_baseline_am/snapshot_epoch_68.ckpt
#'C:/Users/r.rodriguez.outeiral/Documents/Exp_paper/segmentation/new_shifts_8_35_gs/snapshot_epoch44.ckpt'
#'C:/Users/r.rodriguez.outeiral/Documents/Exp_paper/segmentation/and_toy_5_5_t2/snapshot_epoch48.ckpt'
#
#C:/Users/r.rodriguez.outeiral/Documents/Experiments/E2_dist_full/snapshot_epoch118.ckpt
#E:/From_Ritas_machine/diag_unet_am_da/snapshot_epoch312.ckpt
# Parameters
train_model                  = True       # Train the model
test_model                   = False     # Test the model
pred_model                   = False        # Predict using the model
thumbnail                    = False

test_set                     = False

# Batch sizes
batch_size                   =  1              # Batch size
size_train                   = (112, 112, 112)      # Resize the image during training (Height, Width) or None
n_channels                   = 3
nbr_desired_patches          = 32
patch_size                   = 32
border                       = 10            #Border for deepMedic patches
fixed_thr                    = 8     #amount of distance from GT to allow the network to see
moving_thr                   = 0

max_shift=27

#metrics
metrics_dm                   = True

# Data shuffle
shuffle_data                 = False
seed                         = 1924            # Random seed for the shuffle

# Training parameters
optimizer                    = 'adam'       # Optimizer
learning_rate                = 0.001          # Training learning rate
weight_decay                 = 0.              # Weight decay or L2 parameter norm penalty (check)
epochs                       = 250           # Number of epochs during training
snapshots                    = 25         # epochs multiple of this number the model is saved.
cuda                         = True         # If GPU
gpu                          = 3    #If mutli GPU

# Data augmentation
da_norm                      = True        # If normalizing the data according to mean and std
da_hor_flip                  = False    # If Horizontal flip
da_ver_flip                  = True       # If Vertical flip (not use)
da_rotate                    = 10        # Range of the random rotations in degrees
da_deform                    = True         # Level of random deformation we want

# Data aug for both training and testing
da_resize                    = True        #If rescaling (separated)
da_equalize                  = False        # If eq the histogram
random_zeros                 = False

filtering                    = False
# Callbacks
lr_scheduler                 = True
earlyStopping                = True

## checking if you are resizing for the computation of HD.
if da_resize:
    print('Voxel spacing is going to change because of the resizing')
    vox_spacing                  = tuple([275/112*x for x in (0.79323106, 0.7936626, 0.78976499)])
else:
    vox_spacing                  = (0.79323106, 0.7936626, 0.78976499)

