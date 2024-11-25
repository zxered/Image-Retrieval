import cv2 as cv
import os
from pathlib import Path
import numpy as np
import time  # Import the time module to measure execution time

def filter_matches_by_position_and_angle(kp1, kp2, matches, img_width, max_angle_diff=15):
    """
    Filters matches based on angle difference and position:
    - Matches are kept if the angle difference is within max_angle_diff.
    - Keypoints are matched only if both belong to the same half (left/right) of the image.
    """
    filtered_matches = []
    half_width = img_width // 2

    for match in matches:
        # Calculate angle difference
        angle1 = kp1[match.queryIdx].angle
        angle2 = kp2[match.trainIdx].angle
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, 360 - angle_diff)  # Normalize to [0, 180]

        if angle_diff > max_angle_diff:
            continue

        # Filter by position: same half of the image
        x1, _ = kp1[match.queryIdx].pt
        x2, _ = kp2[match.trainIdx].pt

        if ((x1 > half_width) == (x2 > half_width)):
            filtered_matches.append(match)

    return filtered_matches

def precompute_keypoints_and_descriptors(image_folder, orb):
    """
    Precomputes keypoints and descriptors for all images in the specified folder.
    """
    keypoints_descriptors = {}

    for image_path in Path(image_folder).iterdir():
        if image_path.is_file() and image_path.suffix in ['.jpg', '.png']:
            image = cv.imread(str(image_path), cv.IMREAD_COLOR)
            if image is None:
                continue
            red_channel = image[:, :, 2]  # Extract the red channel

            # Detect keypoints and compute descriptors
            kp, des = orb.detectAndCompute(red_channel, None)
            keypoints_descriptors[image_path.name] = (kp, des)

    return keypoints_descriptors

def find_most_similar_image(query_image_name, query_kp_des, db_kp_des, bf, img_width):
    """
    Finds the most similar image in the database based on keypoints and descriptors.
    """
    kp1, des1 = query_kp_des[query_image_name]
    best_match = None
    best_match_score = float('inf')  # Lower score indicates better similarity

    alpha = 1.5  # Weight for scoring matches

    for db_image_name, (kp2, des2) in db_kp_des.items():
        if des1 is None or des2 is None:
            continue

        # Match descriptors
        matches = bf.match(des1, des2)

        # Filter matches by angle and position
        filtered_matches = filter_matches_by_position_and_angle(kp1, kp2, matches, img_width)

        if not filtered_matches:
            continue

        # Compute match score
        total_distance = sum([m.distance for m in filtered_matches])
        num_matches = len(filtered_matches)
        match_score = total_distance / (num_matches ** alpha) if num_matches > 0 else float('inf')

        # Update best match if the score improves
        if match_score < best_match_score:
            best_match_score = match_score
            best_match = db_image_name

    return best_match

def generate_answer_file(query_folder, database_folder, output_file):
    """
    Generates the 'answer.txt' file with the most similar image pairs from query and database folders.
    """
    query_folder = Path(query_folder)
    database_folder = Path(database_folder)
    output_path = Path(output_file)

    # Initialize ORB and BFMatcher
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Precompute keypoints and descriptors
    query_kp_des = precompute_keypoints_and_descriptors(query_folder, orb)
    db_kp_des = precompute_keypoints_and_descriptors(database_folder, orb)

    # Get the width of the first query image
    first_query_image = next(query_folder.glob("*.jpg"), None)
    if first_query_image is None:
        raise FileNotFoundError("Query folder does not contain any .jpg images.")
    img_width = cv.imread(str(first_query_image), cv.IMREAD_COLOR).shape[1]

    # Write results to the output file
    with open(output_path, "w") as f:
        for query_image_name in query_folder.iterdir():
            if query_image_name.is_file() and query_image_name.suffix in ['.jpg', '.png']:
                most_similar_image = find_most_similar_image(
                    query_image_name.name, query_kp_des, db_kp_des, bf, img_width
                )
                f.write(f"query/{query_image_name.name} database/{most_similar_image}\n")

# Paths for input and output
query_folder = "./takehome_dataset/query"
database_folder = "./takehome_dataset/database"
output_file = "./answer.txt"

# Measure execution time
start_time = time.time()  # Start timing
generate_answer_file(query_folder, database_folder, output_file)
end_time = time.time()  # End timing

# Print the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
