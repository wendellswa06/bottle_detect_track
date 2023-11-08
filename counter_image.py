import cv2
import numpy as np

# Create two black images
width = 1520
height = 1080
image1 = np.zeros((height, width, 3), dtype=np.uint8)
image1 = cv2.putText(image1, "count_txt", (45,155), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
image2 = np.ones((height, width, 3), dtype=np.uint8) * 255

# Concatenate the images horizontally
# concatenated_image = np.concatenate((image2, image1), axis=1)
concatenated_image = cv2.copyMakeBorder(image1, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=(0, 0, 255))
print(concatenated_image.shape)
# Display the resulting image
cv2.imshow('Concatenated Image', concatenated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
