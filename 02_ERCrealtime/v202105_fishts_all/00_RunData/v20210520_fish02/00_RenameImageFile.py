####
#### ImageAnalysis for Ecological Reserver Computing
#### OpenCV with Python v6
####

# Prepare folders
#system("mkdir data_image")
#system("mkdir data_temperature")
#system("mkdir raw_image_data")

# Import modules
import numpy as np
import os
import datetime
import subprocess
# XML module
import xml.etree.ElementTree as ET

# Prepare output directory for the next step
os.mkdir("01_BackgroundSubtrOut")

# Rename file
imagefile_folder = "0_process_256"
imagefile_prefix = "Acquisition_Z1_T" # Speficy image file prefix
os.remove("data_image/" + imagefile_folder + "/.DS_Store")

for filename in np.sort(os.listdir("data_image/"+ imagefile_folder)):
  if filename[-4:] == ".jpg":
    # Read xml information
    xml0 = ET.parse('data_image/' + imagefile_folder + '/' + filename[:-4] + '.xml')
    xml1 = xml0.getroot()
    start_date_str0 = xml1[2][0].text
    start_date_str1 = start_date_str0[0:10] + " " + start_date_str0[11:-1]
    datetime_formatted = datetime.datetime.strptime(start_date_str1, "%Y-%m-%d %H:%M:%S")
    start_date = datetime_formatted + datetime.timedelta(hours=9)
    delta_t = round(float(xml1[2][4][2].attrib['DeltaT'])) # xml1[2][5][3].attrib['DeltaT'] until 2021.3.12
    image_date = start_date + datetime.timedelta(seconds=delta_t)
    image_date_str = str(image_date)[0:10] + "_" + str(image_date)[11:13] + "-" + str(image_date)[14:16] + "-" + str(image_date)[17:19]
    
    os.rename("data_image/"+ imagefile_folder + "/" + filename,
              "data_image/"+ imagefile_folder + "/" + image_date_str + ".jpg")
    os.rename("data_image/"+ imagefile_folder + "/" + filename[:-4] + '.xml',
              "data_image/"+ imagefile_folder + "/" + image_date_str + ".xml")

move_command="mv data_image/" + imagefile_folder + "/*.jpg " + "data_image/"
subprocess.call(move_command, shell = True)
