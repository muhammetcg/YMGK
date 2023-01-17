import cv2

# Load the image of the marble block
img = cv2.imread("C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/test_marble/ElazigVisne/_1482_965165.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the image to create a binary image
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# Use the Canny edge detection algorithm to find edges in the binary image
edges = cv2.Canny(thresh,100,200)

# Display the image with the detected edges
cv2.imshow("Original Image",img)
cv2.imshow("Crack Detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
