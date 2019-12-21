import cv2


def show_image(img, title="Image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)

