import matplotlib.pyplot as plt
import sys
from matplotlib.patches import Rectangle
import os


class Annotate(object):
    def __init__(self):
        self.patient_id = str(input("Patient id:"))
        self.laterality = str(input("Laterality:"))
        self.view       = str(input("View:"))

        self.Image_filepath = '/preprocessedData/KUMC-guro_none_gray_original_default/train'
        self.text_filepath = '/preprocessedData/KUMC-guro_none_gray_original_default/roi'

        img_path = '/'.join([
                self.Image_filepath, self.patient_id, '1', self.laterality, self.view]) + '.png'

        self.Image_name = '_'.join([
                self.patient_id, '1', self.laterality, self.view])

        self.im = plt.imread(img_path)
        self.fig, self.ax = plt.subplots(figsize = (10, 20))
        self.implot = self.ax.imshow(self.im, cmap = 'gray')
        self.rect = Rectangle((0,0), 1, 1, facecolor = 'none')
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('key_press_event',  self.key_press)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        plt.show()

    def key_press(self, event):
        k = event.key
        if k == 'm' or k == 'c':
            f = open(os.path.join(self.text_filepath, self.Image_name+'.txt'), 'a')
            f.write(k + '\t')
            f.close()


    def on_press(self, event):

        self.x0 = event.xdata
        self.y0 = event.ydata
        print 'press : ', self.x0, self.y0
        f = open(os.path.join(self.text_filepath, self.Image_name+'.txt'), 'a')
        f.write(str(int(self.x0)) + ',' + str(int(self.y0)))
        f.close()

    def on_release(self, event):

        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()


        print 'release : ', self.x1, self.y1
        f = open(os.path.join(self.text_filepath, self.Image_name+'.txt'), 'a')
        f.write(',' + str(int(self.x1)) + ',' + str(int(self.y1)) + '\n')
        f.close()


while True:
    a = Annotate();
    break
