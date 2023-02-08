import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def concat_dataframes(df_paths):
    dfs = []
    for path in df_paths:
        dfs.append(pd.read_csv(path))
    
    merged_df = pd.concat(dfs)
    return merged_df
        
def visualize_training_plots(metrics_file):
    
    if isinstance(metrics_file, list):
        df = concat_dataframes(metrics_file)
    else:
        df = pd.read_csv(metrics_file)
    
    train_metrics = df.iloc[::2]
    val_metrics = df.iloc[1::2]
    
    epochs = train_metrics.epoch.values
    
    train_loss = train_metrics.loss.values
    train_mean_iou = train_metrics.mean_iou.values
    train_mean_acc = train_metrics.mean_accuracy.values
    
    val_loss = val_metrics.val_loss.values
    val_mean_iou = val_metrics.val_mean_iou.values
    val_mean_acc = val_metrics.val_mean_accuracy.values
    
    fig, axes = plt.subplots(1,3,figsize=(15,6))
    axes[0].plot(epochs,train_loss,'r', label='Training loss')
    axes[0].plot(epochs,val_loss,'b', label='Validation loss')
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss Value')


    axes[1].plot(epochs,train_mean_iou,'r', label='Training mean iou')
    axes[1].plot(epochs,val_mean_iou,'b', label='Validation mean iou')
    axes[1].set_title("Training and validation mean IOU")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean IOU Value')
    axes[1].set_ylim([0, 1])
    
    axes[2].plot(epochs,train_mean_acc,'r', label='Training mean accuracy' )
    axes[2].plot(epochs,val_mean_acc, 'b', label='Validation mean accuracy')
    axes[2].set_title("Training and validation mean Accuracy")
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Mean Accuracy Value')
    axes[2].set_ylim([0, 1])
    plt.legend()
    plt.show()
