from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import cv2 as cv
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import skimage as si
import os

os.chdir(os.path.dirname(__file__))

def cum_hist(image_hist):
    #create an array with all zeros with the same size as image_hist
    cum_hist = np.zeros_like((image_hist))
    
    #add the prev cum_hist value with the current imaage_hist value to get the current cumulative value
    for x in range(0,len(image_hist)):
        if x==0:
            cum_hist[0] = image_hist[0]
        else:
            cum_hist[x] = cum_hist[x-1] + image_hist[x]
    
    return cum_hist

def part1_histogram_compute():

    
    # path = "/home/areez/Desktop/cmput 206/assignment 1/"

    #load the and read input image as grayscale
    test_image_gray = img_as_ubyte(io.imread("test.jpg", True))
 
    #create a new figure to display the images
    fig = plt.figure()
   
    #scikit histogram
    #creat histogram of "test image" using skimage, set bins to 256
    ski, bins = si.exposure.histogram(test_image_gray, nbins=256)
 
    #split up the figure into a 1 by 3 grid
    # place the scikit histogram in the first subplot    
    ax1 = fig.add_subplot(1,3,1)
    #set the title to scikit histogram
    ax1.set_title("scikit histogram")
    #set the xlim to a range of 0 to 256
    ax1.set_xlim(([0,256]))
    ax1.plot(bins, ski)
    
    
   
    #numpy histogram
    #create histogram of "test image using numpy, ", set bins to 256 and range from 0 to 256
    hist,bins = np.histogram(test_image_gray, 256, [0,256])
    
 
    #place the numpy histogram in the second subplot   
    ax2 = fig.add_subplot(1,3,2)
    #set the title to numpy histogram
    ax2.set_title("numpy histogram")
    ax2.plot(hist)
   
    
    #set the number of rows of test image gray to image_height
    #set the number of colums of test image gray to image_width
    image_height = test_image_gray.shape[0]
    image_width = test_image_gray.shape[1]

    #create an empty array(size 256) for the histogram with numpy
    histogram_array = np.zeros([256], np.int32)
    
    #manually compute the histogram by incrementing the iterting through 
    # and incrementing the pixels when needed
    for x in range(0, image_height):
        for y in range(0, image_width):
            histogram_array[test_image_gray[x,y]] += 1
    
   
    #place my histogram in the third subplot
    ax3 = fig.add_subplot(1,3,3)
    #set the title to "my histogram"
    ax3.set_title("my histogram")
    ax3.plot(histogram_array)
    
    #show the figure which contains all three histograms side by side
    plt.show()
  

    """add your code here"""

def part2_histogram_equalization():

    # path = "/home/areez/Desktop/cmput 206/assignment 1/"

    #load and read the image as grayscale and store it in test_image_gray
    test_image_gray = img_as_ubyte(io.imread("test.jpg", True))
   
    #create a new figure to display the image
    fig = plt.figure()
    #display tet_image_gray
    plt.imshow(test_image_gray)
    plt.show()

    #set the number of rows of test image gray to image_height
    #set the number of colums of test image gray to image_width
    image_height = test_image_gray.shape[0]
    image_width = test_image_gray.shape[1]

    #create an empty array(size 256) for the histogram with numpy
    histogram_array = np.zeros([256], np.int32)
    
    #manually compute the histogram
    for x in range(0, image_height):
        for y in range(0, image_width):
            histogram_array[test_image_gray[x,y]] += 1
            
    #set the plots xlim range to 0 to 256
    plt.xlim(0, 256)
    #set the title to "my histogram"
    plt.title("my histogram")
    plt.plot(histogram_array)
    #display the histogram
    plt.show()

   

    #cumulative_histogram
    #use the cum_hist function to get the cumulative histogram of histogram_array
    cum_hist_timage = cum_hist(histogram_array)

    #k = 256 but just did k-1 as k to simplify
    K = 255
    #MN is the images height times the images width
    MN = image_height*image_width
    new_image = test_image_gray.copy()

    #use equalization formula given in class
    for x in range(0, image_height):
        for y in range(0, image_width):
            new_image[x,y] = int((K*(cum_hist_timage[test_image_gray[x,y]]))/MN+0.5)

    #display the new image after histogram equalization
    plt.imshow(new_image)
    plt.show()


    #create an empty array(size 256) for the histogram with numpy
    new_histogram_array = np.zeros([256], np.int32)
    
    #plot new equalized histogram
    for x in range(0, image_height):
        for y in range(0, image_width):
            new_histogram_array[new_image[x,y]] += 1

    #set the xlim to ragne 0 to 256
    plt.xlim(0, 256)
    #set the title to "equalized histogram"
    plt.title("equalized histogram")
    plt.plot(new_histogram_array)
   
    #display the equalized histogram
    plt.show()
    """add your code here"""


def numpy_hisogram(orignal_image):

     # #numpy histogram
    hist,bins = np.histogram(orignal_image, 256, [0,256])
    #set xlim to range 0 to 256
    plt.xlim(0, 256)
    #set ylim to range 0 to 6000
    plt.ylim(0,6000)
    

    return hist

def part3_histogram_comparing():
    """add your code here"""
    
    # path = "/home/areez/Desktop/cmput 206/assignment 1/"

    #read both images and load the image as grayscale
    day_image_gray = img_as_ubyte(io.imread("day.jpg", True))
    night_image_gray = img_as_ubyte(io.imread("night.jpg", True))
  

    #histogram of day.jpg
    #get the numpy histogram of day using the numpy function
    day_hist = numpy_hisogram(day_image_gray)
    
    #histogram of night.jpg
    #get the numpy histogram of night using the numpy function
    night_hist = numpy_hisogram(night_image_gray)

    # #normalize histogram day
    #to normalize divide value by the total sum
    day_norm_hist = day_hist/day_image_gray.size

    # #normalize histogram night
    night_norm_hist = night_hist/night_image_gray.size

    #calculate BC
    #use BC formula given in class
    BC = 0
    for x in range(len(day_hist)):
        BC += np.sqrt((day_norm_hist[x])*(night_norm_hist[x]))

    print("the BC is", BC)


def part4_histogram_matching():
    """add your code here"""

    # path = "/home/areez/Desktop/cmput 206/assignment 1/"

    #read both images and load the image as grayscale
    day_image_gray = cv.imread("day.jpg", cv.IMREAD_GRAYSCALE)
    night_image_gray = cv.imread("night.jpg", cv.IMREAD_GRAYSCALE)
  
    
    #numpy day histogram
    #use the numpy func to get the histogram of day_image_gray
    day_hist = numpy_hisogram(day_image_gray)

    #numpy night histogram
    #use the numpy func to get the histogram of night_image_gray
    night_hist = numpy_hisogram(night_image_gray)


    #set the number of rows of test image gray to image_height
    #set the number of colums of test image gray to image_width
    day_image_height = day_image_gray.shape[0]
    day_image_width = day_image_gray.shape[1]

    #day cumulative histogram
    #use the cum_hist func to get the cumulative histogram of day_hist
    day_cum_hist = cum_hist(day_hist)

    #night cumulative histogram
    #use the cum_hist funct to get the cumulative histogram of night_hist
    night_cum_hist = cum_hist(night_hist)

    #normalize day cumulative histogram
    day_norm_cum_hist = day_cum_hist/day_image_gray.size

    #normalize night cumulative histogram
    night_norm_cum_hist = night_cum_hist/night_image_gray.size

    #create a new empty array that is the same size as day_image_gray
    new_image = np.zeros_like(day_image_gray)
            
    #histogram matching equation
    for i in range(day_image_gray.shape[0]):
        for j in range(day_image_gray.shape[1]):
            a = day_image_gray[i,j]
            a1 = 0
            while night_norm_cum_hist[a1] < day_norm_cum_hist[a]:
                a1 += 1
            new_image[i,j] = a1
    

    #create figure and split into 3 
    fig, ax = plt.subplots(1,3, figsize=(10,10))
    ax[0].imshow(day_image_gray,cmap="gray")
    ax[0].set_title("day image")
    ax[1].imshow(night_image_gray, cmap="gray")
    ax[1].set_title("night image")
    ax[2].imshow(new_image, cmap="gray")
    ax[2].set_title("normalized image")

    #display figure with all images displayed side by side
    plt.show()

    ##################### Question 4 Part B (RGB channel) histogram matching #################################

    #plot day image in RGB
    #convert from BGR to RGB
    day_image_rgb = cv.imread("day.jpg")[:,:,::-1]

    #plot night image in RGB
    #convert from BGR to RGB
    night_image_rgb = cv.imread("night.jpg")[:,:,::-1]

    
    #read images without converting from BGR to RGB
    # used this for the histogram matching
    day_image = cv.imread("day.jpg")
    night_image = cv.imread("night.jpg")
  
    #split and save the rgb channels individually
    day_b, day_g, day_r = cv.split(day_image)
    #day_b = blue day image channel
    #day_r = red day image channel
    #day_g = geen day image channel
    night_b, night_g, night_r = cv.split(night_image)
    #night_b = blue night image channel
    #night_r = red night image channel
    #night_g = geen night image channel

    #red channel histograms for day and night image
    #use the numpy funct to obtain both histograms
    day_red_hist = numpy_hisogram(day_r)
    night_red_hist = numpy_hisogram(night_r)
    
    #blue channel histograms for day and night image
    #use the numpy funct to obtain both histograms
    day_blue_hist = numpy_hisogram(day_b)
    night_blue_hist = numpy_hisogram(night_b)

    #green channel histograms for day and night image
    #use the numpy funct to obtain both histograms
    day_green_hist = numpy_hisogram(day_g)
    night_green_hist = numpy_hisogram(night_g)

    #red channel cumulative and norm for day and night image
    #use the cum_hist funct to obtain both histograms
    #calaulate the normalization of the cum functions
    day_red_cum = cum_hist(day_red_hist)
    day_red_norm_cum_hist = day_red_cum/day_r.size
    night_red_cum = cum_hist(night_red_hist) 
    night_red_norm_cum_hist = night_red_cum/night_r.size

    #green channel cumulative and norm for day and night image
    #use the cum_hist funct to obtain both histograms
    #calaulate the normalization of the cum functions
    day_green_cum = cum_hist(day_green_hist)
    day_green_norm_cum_hist = day_green_cum/day_g.size
    night_green_cum = cum_hist(night_green_hist)
    night_green_norm_cum_hist = night_green_cum/night_g.size

    #blue channel cumulative and norm for day and night image
    #use the cum_hist funct to obtain both histograms
    #calaulate the normalization of the cum functions
    day_blue_cum = cum_hist(day_blue_hist)
    day_blue_norm_cum_hist = day_blue_cum/day_b.size
    night_blue_cum = cum_hist(night_blue_hist)
    night_blue_norm_cum_hist = night_blue_cum/night_b.size



    #red channel
    #use histogram matching
    red_channel = np.zeros_like(day_r)

    for i in range(day_r.shape[0]):
        for j in range(day_r.shape[1]):
            a = day_r[i,j]
            a1 = 0
            while night_red_norm_cum_hist[a1] < day_red_norm_cum_hist[a]:
               
                a1 += 1
            red_channel[i,j] = a1

    #blue channel
    blue_channel = np.zeros_like(day_r)

    for i in range(day_b.shape[0]):
        for j in range(day_b.shape[1]):
            a = day_b[i,j]
            a1 = 0
            while night_blue_norm_cum_hist[a1] < day_blue_norm_cum_hist[a]:
               
                a1 += 1
            blue_channel[i,j] = a1

    #green channel
    green_channel = np.zeros_like(day_r)

    for i in range(day_g.shape[0]):
        for j in range(day_g.shape[1]):
            a = day_g[i,j]
            a1 = 0
            while night_green_norm_cum_hist[a1] < day_green_norm_cum_hist[a]:
               
                a1 += 1
            green_channel[i,j] = a1

    #join the three rgb channels back together and store in rgb
    rgb = np.dstack((red_channel, green_channel, blue_channel))

    #create new figure and split it into a 1 by 3 grid with figsize 10by10
    fig, ax = plt.subplots(1,3, figsize=(10,10))
    ax[0].imshow(day_image_rgb)
    ax[0].set_title("day image")
    ax[1].imshow(night_image_rgb)
    ax[1].set_title("night image")
    ax[2].imshow(rgb)
    ax[2].set_title("normailized image")

    #display the figure which shows all three images side by side
    #takes a couple seconds to load
    plt.show()

if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
