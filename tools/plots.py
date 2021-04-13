import os
import time
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# Plot training hisory
def plot_history(train_loss, train_dice,val_loss, val_dice, save_path):
  
    # Colors (we assume there are no more than 7 metrics):
    colors = ['r', 'g', 'k', 'm', 'c', 'y', 'w']

    # Find the best epoch
    best_train_loss = np.min(train_loss)
    best_train_dice = np.max(train_dice)
    best_val_loss = np.min(val_loss)
    best_val_dice = np.max(val_dice)

    
    # Initialize figure:
    # Axis 1 will be for metrics, and axis 2 for losses.
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    # Plotting
    ax2.plot(train_loss, 'b-', label='{} ({:.3f})'.format('Train Loss', best_train_loss))
    ax1.plot(train_dice, 'r-', label='{} ({:.3f})'.format('Train Dice', best_train_dice))
    ax2.plot(val_loss, 'b--', label='{} ({:.3f})'.format('Val Loss', best_val_loss))
    ax1.plot(val_dice, 'r--', label='{} ({:.3f})'.format('Val Dice', best_val_dice))

    ax1.set_ylim(0,1)
    #ax2.set_ylim(0,1)
    
    # Add title
    plt.title('Model training history')

    # Add axis labels
    ax1.set_ylabel('Metric')
    ax2.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    
    # ??
    fig.tight_layout()

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save fig
    plt.savefig(save_path)

    # Close plot
    plt.close()