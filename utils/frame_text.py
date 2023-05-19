import cv2
import numpy as np


def put_counted_result(img, text1, text2, text3):
    """
    This function is used to put the text with detection information on the video frame.
    """
    lt = 3 or round(0.05* (img.shape[0] + img.shape[1]) / 2) + 1
    ft = max(lt - 1, 4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8

    # Get the eact text size
    text1_size = cv2.getTextSize(text1, font, font_scale, 2)[0]
    text2_size = cv2.getTextSize(text2, font, font_scale, 2)[0]
    text3_size = cv2.getTextSize(text3, font, font_scale, 2)[0]

    # Define (x, y) coordinate of the text        
    text1_x = 10
    text2_x = text1_x + text1_size[0] + 15
    text3_x = text2_x + text2_size[0] + 15
    # text_y = int(img.shape[0] - img.shape[0] * 0.9)
    text_y = text1_size[1] + 10

    # Create white rectangle with the same size with image
    rect_x = text1_x - ft
    rect_y = text_y - text1_size[1] - ft
    rect_width = text3_x + text3_size[0] - text1_x + ft
    rect_height = text1_size[1] + ft * 2
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

    # Finally Put the text
    cv2.putText(img, text1, (text1_x, text_y), font, font_scale, [0, 0, 255], thickness=ft, lineType=cv2.LINE_AA)
    cv2.putText(img, text2, (text2_x, text_y), font, font_scale, [0, 153, 0], thickness=ft, lineType=cv2.LINE_AA)
    cv2.putText(img, text3, (text3_x, text_y), font, font_scale, [255, 0, 0], thickness=ft, lineType=cv2.LINE_AA)
