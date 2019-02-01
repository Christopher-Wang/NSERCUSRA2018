"""
Code taken from: https://matplotlib.org/2.1.2/gallery/animation/image_slices_viewer.html
"""
import numpy as np
import matplotlib.pyplot as plt

#Class to visulize 3D Images
class Visualize3D(object):
    def __init__(self, raws, predictions, masks):
        #Create the figure
        self.fig, self.ax = plt.subplots(1, 3)
        #Connect the values to 
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        #Save the original 3D Images
        self.raws =  raws
        self.predictions =  predictions
        self.masks =  masks
        #Initialize needed variables
        self.slices = raws.shape[0]
        self.ind = 0
        #Create the image subplots
        self.raw = self.ax[0].imshow(self.raws[self.ind], cmap = 'gray', vmin = 0, vmax = 1)
        self.prediction = self.ax[1].imshow(self.predictions[self.ind], cmap = 'gray', vmin = 0, vmax = 1)
        self.mask = self.ax[2].imshow(self.masks[self.ind], cmap = 'gray', vmin = 0, vmax = 1)
        #Display the graph
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        #Create the Labels
        self.fig.suptitle('Slice: %s' % self.ind, fontsize=28)
        self.ax[0].set_xlabel("Raws")
        self.ax[1].set_xlabel('Predictions')
        self.ax[2].set_xlabel('Masks')
        #Set the data
        self.raw.set_data(self.raws[self.ind])
        self.prediction.set_data(self.predictions[self.ind])
        self.mask.set_data(self.masks[self.ind])
        #self.ax[1].imshow(self.predictions[self.ind], cmap = 'gray')
        #self.ax[2].imshow(self.masks[self.ind], cmap = 'gray')
        #Draw the canvass
        self.fig.canvas.draw()

#Class to visulize 3D Images
class Visualize3D_Overlay(Visualize3D):
    def __init__(self, raws, predictions, masks):
        #Create the figure
        self.fig, self.ax = plt.subplots(1, 1)
        #Connect the values to 
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        #Save the original 3D Images
        self.raws =  raws
        self.predictions =  np.ma.masked_where(predictions < 0.5, predictions)
        self.masks =  np.ma.masked_where(masks < 0.5, masks)
        #Initialize needed variables
        self.slices = raws.shape[0]
        self.ind = 0
        #Create the image subplots
        self.raw = self.ax.imshow(self.raws[self.ind], cmap = 'gray', vmin = 0, vmax = 1)
        self.mask = self.ax.imshow(self.masks[self.ind], cmap = 'gray', vmin = 0, vmax = 1, alpha = 1)
        self.prediction = self.ax.imshow(self.predictions[self.ind], cmap = 'cool', vmin = 0, vmax = 1, alpha = 0.4)
        #Display the graph
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        #Create the Labels
        self.fig.suptitle('Slice: %s' % self.ind, fontsize=28)
        #Set the data
        self.raw.set_data(self.raws[self.ind])
        self.prediction.set_data(self.predictions[self.ind])
        self.mask.set_data(self.masks[self.ind])
        #self.ax[1].imshow(self.predictions[self.ind], cmap = 'gray')
        #self.ax[2].imshow(self.masks[self.ind], cmap = 'gray')
        #Draw the canvass
        self.fig.canvas.draw()


class Visualize3D_List(object):
    def __init__(self, images):
        #Create the figure
        self.fig, self.ax = plt.subplots(1, len(images))
        #Connect the values to 
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        #Save the original 3D Images
        self.images = images
        #Initialize needed variables
        self.slices = 88
        self.ind = 0
        self.image = []
        #Create the image subplots
        for i in range(len(images)):
            self.image.append(self.ax[i].imshow(images[i][self.ind], cmap = 'gray', vmin = 0, vmax = 1))
        #Display the graph
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        #Create the Labels
        self.fig.suptitle('Slice: %s' % self.ind, fontsize=28)
        #Set the data
        for i in range(len(self.images)):
             self.image[i].set_data(self.images[i][self.ind])
        #self.ax[1].imshow(self.predictions[self.ind], cmap = 'gray')
        #self.ax[2].imshow(self.masks[self.ind], cmap = 'gray')
        #Draw the canvass
        self.fig.canvas.draw()
