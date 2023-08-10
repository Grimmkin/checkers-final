from detector import get_chessboard_intersections
from utils_visual import *

import cv2, pprint

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# def order_points(pts):
#     # Initialize a list of coordinates that will be ordered
#     # such that the first entry in the list is the top-left,
#     # the second entry is the top-right, the third is the
#     # bottom-right, and the fourth is the bottom-left
#     rect = np.zeros((4, 2), dtype = "float32")

#     # The top-left point will have the smallest sum, whereas
#     # the bottom-right point will have the largest sum
#     s = pts.sum(axis = 1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]

#     # Compute the difference between the points, the
#     # top-right point will have the smallest difference,
#     # whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis = 1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]

#     # Return the ordered coordinates
#     return rect

def returnList(string):
    string = np.array2string(string)

    # Remove the brackets and split the string on whitespace
    numbers_as_strings = string.replace('[', '').replace(']', '').split()

    # Convert the list of strings into a list of integers
    numbers_as_ints = list(map(int, numbers_as_strings))

    return numbers_as_ints

def detect_pawn(tile):
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    
    # Green mask
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Purple mask
    lower_purple = np.array([125, 100, 100])
    upper_purple = np.array([175, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    # Check the presence of color
    if np.max(mask_green) > 0:
        pawn_color = "G"
    elif np.max(mask_purple) > 0:
        pawn_color = "P"
    else:
        pawn_color = ""

    # Create a mask with only the pawn in color and the rest of the image in black
    pawn_mask = cv2.inRange(hsv, lower_green, upper_green) | cv2.inRange(hsv, lower_purple, upper_purple)
    pawn_image = cv2.bitwise_and(tile, tile, mask=pawn_mask)
    
    return pawn_color, pawn_image

def split_into_tiles(image, rows, cols):
    return (image.reshape(rows, image.shape[0]//rows, cols, image.shape[1]//cols, image.shape[2])
            .swapaxes(1,2)
            .reshape(-1, image.shape[0]//rows, image.shape[1]//cols, image.shape[2]))

def transform_board(board_dict):
    # Initialize an 8x8 empty board
    transformed_board = [['---' for _ in range(8)] for _ in range(8)]
    
    # Iterate over original board
    for position, piece in board_dict.items():
        col = ord(position[0]) - ord('A')  # Convert 'A'-'H' to 0-7
        row = int(position[1]) - 1  # Convert '1'-'8' to 0-7
        
        if piece == 'G':  # If there is a piece from player 1 at this position
            transformed_board[col][row] = 'c' + str(row) + str(col)  # 'c' as identifier for player 1
        elif piece == 'P':  # If there is a piece from player 2 at this position
            transformed_board[col][row] = 'b' + str(row) + str(col)  # 'b' as identifier for player 2
            
    return transformed_board


# Load the image
image = cv2.imread("test8.jpg")

# Ensure the image was properly loaded
if image is None:
    print("Error loading image")
else:
    intersections = get_chessboard_intersections(image)
    pprint.pprint(intersections)

    # Assuming intersections is a numpy array of shape (9, 9, 2)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    res_green = cv2.bitwise_and(image,image, mask= mask_green)

    lower_purple = np.array([125, 100, 100])
    upper_purple = np.array([175, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    res_purple = cv2.bitwise_and(image,image, mask= mask_purple)

    intersections = np.apply_along_axis(lambda x: tuple(reversed(x)), 2, intersections)

    top_left = returnList(intersections[0][0])
    top_right = returnList(intersections[0][-1])
    bottom_left = returnList(intersections[-1][0])
    bottom_right = returnList(intersections[-1][-1])

    print("Top Left Corner: ", top_left)
    print("Top Right Corner: ", top_right)
    print("Bottom Left Corner: ", bottom_left)
    print("Bottom Right Corner: ", bottom_right)

    #Prepare vertices
    vertices = np.array([top_right, top_left, bottom_right, bottom_left])

    vertices = order_points(vertices)

    print(vertices)
    vertices = np.int0(vertices)

    output_size = max(np.linalg.norm(vertices[0] - vertices[1]), np.linalg.norm(vertices[1] - vertices[2]), 
                    np.linalg.norm(vertices[2] - vertices[3]), np.linalg.norm(vertices[3] - vertices[0]))
    
    output_size = np.ceil(output_size / 8) * 8

    dst = np.array([[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(np.float32(vertices), dst)

    warped = cv2.warpPerspective(image, M, (int(output_size), int(output_size)))

    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = ['1', '2', '3', '4', '5', '6', '7', '8']
    tiles = split_into_tiles(warped, 8, 8)
    tile_status = {}

    for i, tile in enumerate(tiles):
        row = i // 8
        col = i % 8
        if (row % 2 == 0 and col % 2 == 0) or (row % 2 == 1 and col % 2 == 1):
            pawn_color, image = detect_pawn(tile)
            tile_status[rows[row] + cols[col]] = pawn_color

            # Display the processed tile
            #cv2.imshow(f'Tile-{i} {rows[row] + cols[col]}', image)

    print(tile_status)

    pprint.pprint(transform_board(tile_status))

    image = draw_intersections_on_image(image, intersections)

    cv2.imshow("Intersections", image)
    # cv2.imshow('image',image)
    # cv2.imshow('green mask',res_green)
    # cv2.imshow('purple mask',res_purple)
    cv2.imshow('warped', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image = draw_intersections_on_image(image, intersections)