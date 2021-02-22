# Cmput-206-lab-1
Assignment 1: scikit-image - Histograms

You are provided with a single source file called A1_submission.py along with 3 images to be used as input in your code. You need to complete the four functions indicated there, one for each part. You can add any other functions or other code you want to use but they must all be in this same file.

You need to submit only the completed A1_submission.py.

Part I (20%): Complete function part1_histogram_compute() to accomplish the following:

Read the grayscale image called test.jpg
Write your own code to compute a 256-bin gray scale histogram of the image. You cannot use built in histogram functions from any library (e.g. numpy.histogram, scipy.stats.histogram, skimage.exposure.histogram, opencv.calcHist, etc) for this.
Plot the histogram.
Also, call Skimage and Numpy histogram functions to compute 256-bin histograms for the same image. Plot both histograms side by side with yours to show that they are identical.

Part II (30%): Histogram equalization:

Complete function part2_histogram_equalization() to perform grayscale histogram equalization on the same test.jpg image used in the last part. You need to plot the original image, its histogram, the image after histogram equalization and its histogram. You are not allowed to use the Skimage functions, i.e.,exposure.histogram, exposure.equalize_hist or any equivalent functions from any other library for this part.

Part III (10%): Histogram comparing:

Complete function part3_histogram_comparing() to compare the histograms of two images day.jpg and night.jpg. You will need to read both images, convert them to grayscale, compute their histograms and print the Bhattacharyya Coefficient of the two histograms. You can use Skimage or other histogram functions to compute the histograms.

Part IV (40%): Histogram matching:

Complete function part4_histogram_matching() to match the histograms of the same two images day.jpg and night.jpg from part 3. You can use Skimage or other histogram function to compute the histograms.

(a) (30%) Grayscale:

Read both images, convert them to grayscale, and match the histogram of day image to that of the night image to generate a new grayscale image that should be a darker version of the day image. Show the grayscale day, night and matched day images side by side.

(b) (10%) RGB: 

Repeat the grayscale histogram matching process from part (a) with each channel of the 2 images and put together the resultant matched channels into an RGB image.You can also use the single intensity mapping obtained from the grayscale images in (a) to match each of the three channels as suggested in the third tutorial below. Show the RGB day, night and matched day images side by side.
