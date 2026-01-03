import cv2
import math
import os
import numpy as np

def sort_keypoints_by_response(keypoints, descriptors):
    # Match each keypoint with its original subscript
    indexed_kp = list(enumerate(keypoints))
    # Sort by response in descending order
    indexed_kp.sort(key=lambda x: x[1].response, reverse=True)

    # sorted keypoints
    sorted_keypoints = [kp for (_, kp) in indexed_kp]
    # Sorted descriptors (according to the same subscript order)
    sorted_descriptors = descriptors[[idx for (idx, _) in indexed_kp], :]
    return sorted_keypoints, sorted_descriptors

def is_correct(match, keypoints1, keypoints2, threshold=5.0):
    kp1 = keypoints1[match.queryIdx].pt  # (x1, y1)
    kp2 = keypoints2[match.trainIdx].pt  # (x2, y2)
    dist = math.sqrt((kp1[0] - kp2[0]) ** 2 + (kp1[1] - kp2[1]) ** 2)
    return dist < threshold

def process_all_images(input_dir1, input_dir2, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files1 = os.listdir(input_dir1)

    for filename in files1:
        path1 = os.path.join(input_dir1, filename)
        path2 = os.path.join(input_dir2, filename)

        if os.path.isfile(path1) and os.path.isfile(path2):
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)

            if img1 is None or img2 is None:
                print(f"Warning: Unable to read image{filename}，Skipped.")
                continue

            orb = cv2.ORB_create(nfeatures=400)
            keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

            # If the descriptor is empty, it will cause BFMatcher.match to fail, so a judgment needs to be made
            if descriptors1 is None or descriptors2 is None:
                print(f"Warning: The descriptor for image {filename} is empty, skipped.")
                continue

            # Re-check the shape of the truncated descriptor
            if descriptors1.size == 0 or descriptors2.size == 0:
                print(f"Warning: The descriptor for image {filename} is empty, skipped.")
                continue

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)

            correct_matches = []
            wrong_matches   = []
            for m in matches:
                if is_correct(m, keypoints1, keypoints2, threshold=50.0):
                    correct_matches.append(m)
                else:
                    wrong_matches.append(m)

            for kp in keypoints1:
                kp.size = 20
            for kp in keypoints2:
                kp.size = 20

            # Draw the error matches first (in red)
            res = cv2.drawMatches(
                img1, keypoints1,
                img2, keypoints2,
                wrong_matches,
                None,
                matchColor=(0,0,255),
                singlePointColor=(0,255,0),
                matchesThickness=2,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, res)
            print(f"The matching results have been processed and saved:{output_path}")
        else:
            print(f"Warning:{filename}does not correspond completely in two folders and has been skipped.")

if __name__ == "__main__":
    input_dir1 = "./input_dir1"
    input_dir2 = "./input_dir2"
    output_dir = "./matches"
    process_all_images(input_dir1, input_dir2, output_dir)
