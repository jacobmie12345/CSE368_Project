import gymnasium as gym
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_frame(frame, scale=0.5):
    height, width, _ = frame.shape
    new_size = (int(width * scale), int(height * scale))
    resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return resized_frame

def segment_color(frame, lower_bound, upper_bound):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    return mask.astype(np.uint8)

def detect_mario(frame):
    mario_lower = (0, 150, 50)
    mario_upper = (10, 255, 255)
    mario_mask = segment_color(frame, mario_lower, mario_upper)
    mario_contours, _ = cv2.findContours(mario_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in mario_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 3:
            return (x, y)
    return None

def detect_barrels(frame):
    barrel_lower = (20, 100, 100)
    barrel_upper = (30, 255, 255)
    barrel_mask = segment_color(frame, barrel_lower, barrel_upper)
    barrel_contours, _ = cv2.findContours(barrel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    barrels = set()
    for contour in barrel_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if y >= 15 and w > 3:
            barrels.add((x, y))
    return barrels

def detect_daisy(frame):
    daisy_lower = (90, 50, 50)
    daisy_upper = (120, 255, 255)
    daisy_mask = segment_color(frame, daisy_lower, daisy_upper)
    daisy_contours, _ = cv2.findContours(daisy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in daisy_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 20 and h <= 20:
            return (x, y)
    return None

def detect_dk(frame):
    dk_lower = (5, 50, 50)
    dk_upper = (20, 255, 200)
    dk_mask = segment_color(frame, dk_lower, dk_upper)
    dk_contours, _ = cv2.findContours(dk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in dk_contours:
        x, y, w, h = cv2.boundingRect(contour)
        return (x, y)
    return None

def detect_ladders(frame):
    ladder_lower = (130, 50, 50)
    ladder_upper = (160, 255, 255)
    ladder_mask = segment_color(frame, ladder_lower, ladder_upper)
    ladder_contours, _ = cv2.findContours(ladder_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ladders = set()
    for contour in ladder_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > h and w < 5:
            ladders.add((x, y))
    return ladders

def stack_frames(frame, stack, stack_size=4, scale=0.5):
    height, width, _ = frame.shape
    new_size = (int(width * scale), int(height * scale))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, new_size, interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame / 255.0 
    processed_frame = np.expand_dims(normalized_frame, axis=0)
    stack.append(processed_frame)
    while len(stack) < stack_size:
        stack.append(processed_frame)
    stacked_frames = np.concatenate(list(stack), axis=0)
    return stacked_frames

def visualize_frame(frame):
    frame_visual = frame.copy()
    # Detect features
    mario_position = detect_mario(frame)
    barrel_positions = detect_barrels(frame)
    daisy_position = detect_daisy(frame)
    dk_position = detect_dk(frame)
    ladder_positions = detect_ladders(frame)

    def draw_rectangle(x, y, color):
        top_left = (max(0, x - 5), max(0, y - 5))
        bottom_right = (min(frame.shape[1] - 1, x + 5),min(frame.shape[0] - 1, y + 5))
        cv2.rectangle(frame_visual, top_left, bottom_right, color, 1)

    draw_rectangle(mario_position[0], mario_position[1], (255, 0, 0))
    draw_rectangle(daisy_position[0], daisy_position[1], (0, 0, 255))
    draw_rectangle(dk_position[0], dk_position[1], (255, 165, 0))

    for x, y in barrel_positions:
        draw_rectangle(x, y, (0, 255, 0))
    for x, y in ladder_positions:
        draw_rectangle(x, y, (128, 0, 128))

    plt.imshow(frame_visual)
    plt.title("Detected Features")
    plt.axis("off")
    plt.show()