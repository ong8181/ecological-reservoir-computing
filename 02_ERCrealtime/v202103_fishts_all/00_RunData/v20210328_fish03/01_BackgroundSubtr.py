from ij import IJ
from ij.io import DirectoryChooser
import os

def Analysis(imagepath, outDir, filename):
    imp = IJ.openImage(imagepath)
    savefilepath = os.path.splitext(imagepath)[0]
    #####################################################	
    IJ.run(imp, "8-bit", "")
    IJ.run(imp, "Subtract Background...", "rolling=15")
    IJ.saveAs(imp, "Jpeg", "%s/%s" % (outDir, filename))
    #####################################################
    imp.close()


srcDir = DirectoryChooser("Choose input folder").getDirectory()
IJ.log("directory: " + srcDir)
outDir = DirectoryChooser("Choose output folder").getDirectory()
IJ.log("directory: " + outDir)

root = os.path.dirname(srcDir)
for filename in os.listdir(root):
	if filename.endswith(".jpg"):
		path = os.path.join(root, filename)
		IJ.log(path)
		Analysis(path, outDir, filename)


IJ.log("Finish")
