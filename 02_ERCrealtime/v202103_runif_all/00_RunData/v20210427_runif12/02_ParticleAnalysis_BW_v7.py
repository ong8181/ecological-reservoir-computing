####
#### ImageAnalysis for Ecological Reserver Computing
#### OpenCV with Python BW
####

## Reference
## https://stackoverflow.com/questions/51621684/how-to-calculate-nucleus-amount-of-cell
## To get HSV values easily, use https://www.color-site.com/image_pickers
##

# Import modules
import numpy as np
import pandas as pd
import scipy
import cv2
import os
import datetime
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.segmentation import watershed as skwater
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 100)

# Create output folder
output_folder="02_ParticleAnalysisOut"
os.makedirs(output_folder)

# Define ShowImage function
def ShowImage(title,img,ctype): # ----------------- #
  if ctype=='bgr':
    b,g,r = cv2.split(img) # get b,g,r
    rgb_img = cv2.merge([r,g,b]) # switch it to rgb
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.title(title)
  plt.show()
# ------------------------------------------------- #

# --------------------------------------------------- #
# ------------------ Main loop ---------------------- #
# --------------------------------------------------- #
# User-defined paramters
calc_distribution_index = False
max_cell_area = 1000
min_cell_area = 60
image_folder = "data_image"
image_folder_bw = "01_BackgroundSubtrOut"
os.remove(image_folder_bw + "/.DS_Store") # Delete an unnecessary hidden file

for filename in np.sort(os.listdir(image_folder_bw + "/")):
  # Read xml information
  if filename[-3:]=="jpg":
    image_date_str = filename[0:10] + " " + filename[11:13] + ":" + filename[14:16] + ":" + filename[17:19]
    
    # Read in image
    img0 = cv2.imread(image_folder + '/' + filename)
    gray = cv2.imread(image_folder_bw + "/" + filename)
    hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    #ShowImage('img', hsv,'gray')
    
    # Convert pixel space to an array of triplets. These are vectors in 3-space.
    Z = np.float32(hsv.reshape((-1,1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Perform the k-means transformation. What we get back are:
    # Define the number of clusters to find
    K = 2
    #*Centers: The coordinates at the center of each 3-space cluster
    #*Labels: Numeric labels for each cluster
    #*Ret: A return code indicating whether the algorithm converged, &c.
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Adjust center if center[1] is too small
    # Because a too small value implies no or a small number of cells is in the image
    if float(max(center)) < 80:
      center[np.argmax(center)] = 80
      label[Z >= (center[np.argmax(center)] - 60)] = np.argmax(center)
      label[Z < center[np.argmax(center)] - 60] = np.argmin(center)
    # Produce an image using only the center colours of the clusters
    center = np.uint8(center)
    khsv   = center[label.flatten()]
    khsv   = khsv.reshape((hsv.shape))
    # Reshape labels for masking
    label = label.reshape(hsv.shape)
    #ShowImage('img',label,'gray')
    #ShowImage('img',khsv,'gray')
    
    # Remove noise by eliminating single-pixel patches
    khsv2 = np.uint8(khsv)
    kernel  = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(khsv2, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # Normalize data
    opening = (opening - opening.min()) / (opening.max() - opening.min())
    opening = np.uint8(opening)
    #ShowImage('img',opening,'gray')
    
    # Identify areas which are surely foreground
    h_fraction = 0.1
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    maxima = extrema.h_maxima(dist, h_fraction*dist.max())
    # Dilate the maxima so we can see them
    maxima_dilate = cv2.dilate(maxima, kernel, iterations=2)
    # Finding unknown region
    binary_w_maxima = cv2.subtract(opening, maxima_dilate)
    #ShowImage('img',maxima_dilate,'gray')
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(maxima_dilate)
    # Add one to all labels so that sure background is 1, not 0.
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[binary_w_maxima==np.max(binary_w_maxima)] = 0
    markers = skwater(-dist, markers, watershed_line = True)
    markers_detected = markers.copy()
    #ShowImage('img',markers[1150:1250,1450:1650],'gray')
    
    # ------------------------------------------------------ #
    # Single grid version -------------------------------- #
    grid_n = 1
    markers_grid_i = markers.copy()
    markers_grid_i_detected = markers.copy()
    #ShowImage('img',markers_grid_i_detected,'gray')
    grid_cell_count = 0
    grid_cell_area = 0
    grid_cell_mean_area = 0
    center_coord = np.empty((0,2), int)
    if len(np.unique(markers_grid_i)) > 0:
      for l in np.unique(markers_grid_i):
          if l==0 or l==1: # Watershed line or Background
              continue
          else:
            grid_cell_area_tmp = np.sum(markers_grid_i==l)
            if grid_cell_area_tmp < max_cell_area and grid_cell_area_tmp > min_cell_area:
                # Count total number of cells and calculate total cell area
                grid_cell_area += grid_cell_area_tmp
                grid_cell_count += 1
                if not calc_distribution_index:
                  # Calculate the center of cell
                  center_i = np.mean(np.argwhere(markers_grid_i==l), axis = 0)
                  center_x = int(round(center_i[0]))
                  center_y = int(round(center_i[1]))
                  center_coord_tmp = np.array([center_x, center_y])
                  center_coord = np.vstack([center_coord, center_coord_tmp])
                # Replace marker ID with -10 for visualization
                markers_grid_i_detected[markers_grid_i==l] = -10
    # Calculate mean cell area
    if grid_cell_count > 0:
      grid_cell_mean_area = grid_cell_area/grid_cell_count
    else:
      grid_cell_mean_area = np.nan
    # Calculate distance between centers
    if grid_cell_count < 2 or not calc_distribution_index:
      min_dist = min_dist_mean = min_dist_sd = min_dist_skew = min_dist_kurt = np.nan
    else:
      center_difs = np.expand_dims(center_coord, axis=1) - np.expand_dims(center_coord, axis=0)
      center_dist = np.sqrt(np.sum(center_difs ** 2, axis=-1))
      center_dist[np.eye(center_dist.shape[0], dtype = bool)] = np.nan
      min_dist = np.nanmin(center_dist, axis = 1)
      min_dist_mean = np.mean(min_dist)
      min_dist_sd = np.std(min_dist)
      min_dist_skew = scipy.stats.skew(min_dist)
      min_dist_kurt = scipy.stats.kurtosis(min_dist)
    # Summarize grid information
    textFile = open(output_folder + "/0_results.txt","a")
    textFile.write(filename +
                   "; date_time: " + image_date_str +
                   "; grid_n: {}".format(grid_n) +
                   "; count: {}".format(grid_cell_count) +
                   "; total_area: {}".format(grid_cell_area) +
                   "; mean_area: {:.5g}".format(grid_cell_mean_area) +
                   "; NNdist_mean: {:.4g}".format(min_dist_mean) +
                   "; NNdist_sd: {:.4g}".format(min_dist_sd) +
                   "; NNdist_skew: {:.4g}".format(min_dist_skew) +
                   "; NNdist_kurt: {:.4g}".format(min_dist_kurt) +
                   "\n")
    textFile.close()
    markers_reconst = markers_grid_i_detected
    
    # Marked output image
    imgout = img0.copy()
    if grid_cell_count != 0:
      imgout[markers == 0] = [255, 255, 255] # Label the watershed_line
      imgout[markers_reconst == -10] = [100, 100, 255] # Fill the detected area
    #ShowImage('imgout',imgout,'bgr')
    
    cv2.imwrite(output_folder + "/" + filename[:-4] + "_analyzed.jpg", imgout)

# --------------------------------------------------- #
# --------------------- End ------------------------- #
# --------------------------------------------------- #
