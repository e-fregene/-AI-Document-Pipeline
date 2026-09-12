import cv2
import numpy as np
from PIL import Image

#Path to image
img_path = "Tom&Jerry.jpg"

"""OpenCV"""

#Load  image
img_cv = cv2.imread(img_path)

#Resize image to 300x300 pixels
resized = cv2.resize(img_cv, (300, 300))
#cv2.imshow("Resized OpenCV Image", resized)
#cv2.destroyAllWindows()


# Crop region (x: 50-250, y: 100-300)
cropped = img_cv[100:300, 50:250]

#cv2.imshow("Cropped Image", cropped)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Flipping Horizontal;
flipped = cv2.flip(img_cv, 1) #1=horizontal, 0 vertical, -1 both

#cv2.imshow("Flipped Image", flipped)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Rotate image
rotated = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
#cv2.imshow("Rotated Image", rotated)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#Color conversion
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


#cv2.imshow("OpenCV Image", img_cv)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#Saving Images
cv2.imwrite("processed_image.jpg", gray)
cv2.imwrite("processed_image.jpg", cropped)
cv2.imwrite("processed_image.jpg", resized)



"""Pillow (PIL)""" 
#Load image
img_pil = Image.open(img_path)
img_pil.save("processed_T&J.png") #save as PNG

img_pil.show()


"""NumPy"""

#Convert image -> array, print shape
img_np = np.array(img_pil)
print("NumPy Image Shape:", img_np.shape)



"""
Noise reduction
"""

# Load image
img = cv2.imread("document.jpg")

# Apply Gaussian Blur
gaussian = cv2.GaussianBlur(img, (5,5), 0)

# Apply Median Filtering
median = cv2.medianBlur(img, 5)

# Apply Bilateral Filtering
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Gaussian Blur", gaussian)
cv2.imshow("Median Filtering", median)
cv2.imshow("Bilateral Filtering", bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()