import cv2

# Load the image of the marble block
img = cv2.imread("C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/test_marble/ElazigVisne/_1482_965165.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the image to create a binary image
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Calculate the percentage of white pixels in the binary image
white_pixels = cv2.countNonZero(thresh)
total_pixels = img.shape[0] * img.shape[1]
percent_white = (white_pixels / total_pixels) * 100

# Print the result
print("Homojenlik:", percent_white)
