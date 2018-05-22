#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import dataset, models, helper


def save_image(figname, image, mask_true, mask_pred, alpha=0.3):
    mask_true=np.squeeze(mask_true)
    mask_pred=np.squeeze(mask_pred)
    cmap = plt.cm.gray
    plt.figure(figsize=(12, 3.75))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)
    plt.imshow(mask_pred, cmap=cmap, alpha=alpha)
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)
    plt.imshow(mask_true, cmap=cmap, alpha=alpha)
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

def sorensen_dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return 2*intersection / (np.sum(y_true) + np.sum(y_pred))

def jaccard(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    return intersection / union

def compute_statistics(model, generator, steps_per_epoch, return_images=False):
    dices = []
    jaccards = []
    predictions = []
    for i in range(steps_per_epoch):
        images, masks_true = next(generator)
        # Normally: masks_pred = model.predict(images)
        # But some models cannot handle large batch size
        masks_pred = np.concatenate([model.predict(image[None,:,:,:]) for image in images])
        for mask_true, mask_pred in zip(masks_true, masks_pred):
            y_true = mask_true[:,:].astype('uint8')
            y_pred = np.round(mask_pred[:,:]).astype('uint8')
            dices.append(sorensen_dice(y_true, y_pred))
            jaccards.append(jaccard(y_true, y_pred))
        if return_images:
            for image, mask_true, mask_pred in zip(images, masks_true, masks_pred):
                predictions.append((image[:,:,0], mask_true[:,:], mask_pred[:,:]))
    print("Dice:    {:.3f} ({:.3f})".format(np.mean(dices), np.std(dices)))
    print("Jaccard: {:.3f} ({:.3f})".format(np.mean(jaccards), np.std(jaccards)))
    return dices, jaccards, predictions

def main():
    # Sort of a hack:
    # args.outfile = file basename to store train / val dice scores
    # args.checkpoint = turns on saving of images
    with open(os.getcwd() + '/train_config.json') as data_file:
        parameters = json.load(data_file)


    print("Loading dataset...")
    augmentation_args = {
        'rotation_range': parameters["rotation_range"],
        'width_shift_range': parameters["width_shift_range"],
        'height_shift_range': parameters["height_shift_range"],
        'shear_range': parameters["shear_range"],
        'zoom_range': parameters["zoom_range"],
        'fill_mode': parameters["fill_mode"],
        'alpha': parameters["alpha"],
        'sigma': parameters["sigma"],
        'horizontal_flip': parameters["horizontal_flip"],
        'vertical_flip': parameters["vertical_flip"],
    }
    train_generator, train_steps_per_epoch, \
    val_generator, val_steps_per_epoch = dataset.create_generators(
        data_dir=os.getcwd(),
        batch_size=parameters["batch_size"],
        validation_split=parameters["validation_split"],
        shuffle_train_val=parameters["shuffle_train_val"],
        shuffle=parameters["shuffle"],
        seed=None,
        normalize_images=True,
        augment_training=parameters["augment_training"],
        augment_validation=parameters["augment_validation"],
        augmentation_args=augmentation_args)


    # get image dimensions from first batch
    images, masks = next(train_generator)
    _, height, width, channels = images.shape
    _, _, _, classes = masks.shape

    print("Building model...")

    m = models.get_unet_8(height, width)

    m.load_weights(os.path.join(os.getcwd(),'checkpoints', parameters['model_to_load']))

    outfile='prediction_scores'

    checkpoint=os.path.join(os.getcwd(),'save_test_images')

    print("Training Set:")
    train_dice, train_jaccard, train_images = compute_statistics(
        m, train_generator, train_steps_per_epoch,
        return_images=checkpoint)
    print()
    print("Validation Set:")
    val_dice, val_jaccard, val_images = compute_statistics(
        m, val_generator, val_steps_per_epoch,
        return_images=checkpoint)

    if outfile:
        train_data = np.asarray([train_dice, train_jaccard]).T
        val_data = np.asarray([val_dice, val_jaccard]).T
        np.savetxt(outfile + ".train.val.txt", train_data)
        np.savetxt(outfile + ".val.val.txt", val_data)

    if checkpoint:
        print("Saving images...")
        for i,dice in enumerate(train_dice):
            image, mask_true, mask_pred = train_images[i]
            figname = "train-{:03d}-{:.3f}.png".format(i, dice)
            save_image(figname, image, mask_true, np.round(mask_pred))
        for i,dice in enumerate(val_dice):
            image, mask_true, mask_pred = val_images[i]
            figname = "val-{:03d}-{:.3f}.png".format(i, dice)
            save_image(figname, image, mask_true, np.round(mask_pred))
    print('Finished')