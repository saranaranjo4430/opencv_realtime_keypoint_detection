import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


'''
1. Loading the Image and Test Video Capture
'''
# Load the marker image and convert to grayscale
print("Load and display image")
marker_image_path = '/Users/saranaranjo/VAR/BCV/Project/assets/carte_etudiante.jpg'
marker_image = cv2.imread(marker_image_path)
if marker_image is None:
    print("Error: Marker image not found.")
    exit()
marker_image_gray = cv2.cvtColor(marker_image, cv2.COLOR_BGR2GRAY)


# Display the marker image for verification
plt.imshow(marker_image_gray, cmap='gray')
plt.title("Marker Image")
plt.axis("off")
plt.show()


'''
2. Keypoint Detector and Descriptor Matcher Initialization
'''
# Initialize the keypoint detector and matcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)



'''
3. Keypoint Detection and Descriptor Visualization on the Marker Image
'''
# Show keypoints and descriptors on the image
print("Show keypoints and descriptors on the image")
# Detect keypoints and descriptors in the marker image
keypoints_marker, descriptors_marker = sift.detectAndCompute(marker_image_gray, None)
if descriptors_marker is None:
    print("Error: No descriptors found in the marker image.")
    exit()

# Vérifier le résultat
print(f"Number of keypoints detected on the marker_image : {len(keypoints_marker)}")
print(f"Dimensions of the descriptors : {descriptors_marker.shape if descriptors_marker is not None else 'None'}")

# Dessiner les keypoints sur l'image
marker_with_keypoints = cv2.drawKeypoints(
    marker_image, keypoints_marker, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Afficher l'image avec les keypoints
plt.figure(figsize=(10, 8))
plt.imshow(marker_with_keypoints, cmap='gray')
plt.title("Keypoints de l'image marker")
plt.axis("off")
plt.show()




'''
4. Real-Time Keypoint Detection and Descriptor Visualization on the Camera Feed
'''
# Show keypoints and descriptors on the camera
print("Show keypoints and descriptors on the camera")
# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

try:
    while cap.isOpened():
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break
        
        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

        # Print keypoints and descriptors info
        print(f"Number of keypoints detected: {len(keypoints_frame)}")
        if descriptors_frame is not None:
            print(f"Descriptors shape: {descriptors_frame.shape}")
        else:
            print("No descriptors found.")

        # Draw the keypoints on the frame
        frame_with_keypoints = cv2.drawKeypoints(
            frame_gray, keypoints_frame, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Display the frame with keypoints
        cv2.imshow('Keypoints and Descriptors', frame_with_keypoints)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    

'''
5. Keypoint Matching Between Marker Image and Live Camera Feed
'''
# Show two images side by side 
print("Show two images side by side")
# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

try:
    while cap.isOpened():
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break
        
        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors in the current frame
        keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)
        
        if descriptors_frame is not None:
            # Match descriptors between the marker image and the frame
            matches = bf.knnMatch(descriptors_marker, descriptors_frame, k=2)

            # Apply Lowe's ratio test
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            # Draw matches (optional for debugging)
            frame_with_matches = cv2.drawMatches(
                marker_image, keypoints_marker, frame, keypoints_frame, 
                good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # Show the frame with matches
            cv2.imshow('Descriptor Matching', frame_with_matches)
        else:
            # Show the live video if no descriptors are found
            cv2.imshow('Descriptor Matching', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()







'''
6. Homography Estimation and Marker Overlay with Inlier Visualization
'''
# Replace the image on the video
print("Replace the image on the video, inliers")
# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors in the frame
        keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

        if descriptors_frame is not None:
            # Match descriptors between marker and frame
            matches = bf.knnMatch(descriptors_marker, descriptors_frame, k=2)

            # Apply Lowe's ratio test
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) >= 4:
                # Compute homography and get inliers
                src_pts = np.float32([keypoints_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # Warp the marker image
                    h, w = marker_image_gray.shape
                    warped_marker = cv2.warpPerspective(marker_image, M, (frame.shape[1], frame.shape[0]))

                    # Overlay the marker onto the frame
                    frame_with_overlay = cv2.addWeighted(frame, 0.7, warped_marker, 0.3, 0)

                    # Visualize inliers
                    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i, 0]]
                    frame_with_inliers = cv2.drawMatches(
                        marker_image, keypoints_marker, frame, keypoints_frame,
                        inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

                    # Resize the inlier matches to match the dimensions of the overlay
                    frame_with_inliers_resized = cv2.resize(
                        frame_with_inliers, (frame_with_overlay.shape[1], frame_with_overlay.shape[0])
                    )

                    # Concatenate the frames side by side
                    combined_frame = np.hstack((frame_with_overlay, frame_with_inliers_resized))

                    # Display the side-by-side visualization
                    cv2.imshow('Homography Estimation and Marker Overlay with Inlier Visualization', combined_frame)

                    # Inlier ratio
                    inlier_ratio = np.sum(mask) / len(good_matches)
                    print(f"Inlier Ratio: {inlier_ratio:.2f}")
            else:
                cv2.putText(frame, "Not enough matches", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Descriptor Matching', frame)
        else:
            cv2.imshow('Descriptor Matching', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()











'''
7. Enhanced Marker Replacement with Post-Warping Filters
'''
print("Replace the image on the video with post-warping filters")
# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors in the frame
        keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

        if descriptors_frame is not None:
            # Match descriptors between marker and frame
            matches = bf.knnMatch(descriptors_marker, descriptors_frame, k=2)

            # Apply Lowe's ratio test
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) >= 4:
                # Compute homography and get inliers
                src_pts = np.float32([keypoints_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # Warp the marker image
                    h, w = marker_image_gray.shape
                    warped_marker = cv2.warpPerspective(marker_image, M, (frame.shape[1], frame.shape[0]))

                    # **Post-Warping Filters**
                    # 1. Gaussian Blur
                    warped_marker = cv2.GaussianBlur(warped_marker, (5, 5), 0)

                    # 2. Adjust Brightness and Contrast
                    alpha = 1.2  # Contrast factor
                    beta = 20    # Brightness offset
                    warped_marker = cv2.convertScaleAbs(warped_marker, alpha=alpha, beta=beta)

                    # 3. Edge Enhancement (optional)
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
                    warped_marker = cv2.filter2D(warped_marker, -1, kernel)

                    # Overlay the marker onto the frame
                    frame_with_overlay = cv2.addWeighted(frame, 0.7, warped_marker, 0.3, 0)

                    # Visualize inliers
                    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i, 0]]
                    frame_with_inliers = cv2.drawMatches(
                        marker_image, keypoints_marker, frame, keypoints_frame,
                        inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

                    # Resize the inlier matches to match the dimensions of the overlay
                    frame_with_inliers_resized = cv2.resize(
                        frame_with_inliers, (frame_with_overlay.shape[1], frame_with_overlay.shape[0])
                    )

                    # Concatenate the frames side by side
                    combined_frame = np.hstack((frame_with_overlay, frame_with_inliers_resized))

                    # Display the side-by-side visualization
                    cv2.imshow('Enhanced Marker Replacement with Post-Warping Filters', combined_frame)

                    # Inlier ratio
                    inlier_ratio = np.sum(mask) / len(good_matches)
                    print(f"Inlier Ratio: {inlier_ratio:.2f}")
                else:
                    print("Homography could not be computed.")
            else:
                cv2.putText(frame, "Not enough matches", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Descriptor Matching', frame)
        else:
            cv2.imshow('Descriptor Matching', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()















'''
Conclusions
'''
# Initialize data storage for analysis
results = []

print("Replace the image on the video with post-warping filters")
# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Start processing time
        start_time = time.time()

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors in the frame
        keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

        if descriptors_frame is not None:
            # Match descriptors between marker and frame
            matches = bf.knnMatch(descriptors_marker, descriptors_frame, k=2)

            # Apply Lowe's ratio test
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) >= 4:
                # Compute homography and get inliers
                src_pts = np.float32([keypoints_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # Compute reprojection error (homography accuracy)
                    projected_pts = cv2.perspectiveTransform(src_pts, M)
                    reprojection_error = np.mean(np.linalg.norm(projected_pts - dst_pts, axis=2))

                    # Compute inlier ratio
                    inlier_ratio = np.sum(mask) / len(good_matches)

                    # Warp the marker image
                    h, w = marker_image_gray.shape
                    warped_marker = cv2.warpPerspective(marker_image, M, (frame.shape[1], frame.shape[0]))

                    # Post-warping filters
                    warped_marker = cv2.GaussianBlur(warped_marker, (5, 5), 0)
                    warped_marker = cv2.convertScaleAbs(warped_marker, alpha=1.2, beta=20)
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    warped_marker = cv2.filter2D(warped_marker, -1, kernel)

                    # Overlay the marker onto the frame
                    frame_with_overlay = cv2.addWeighted(frame, 0.7, warped_marker, 0.3, 0)

                    # Visualize inliers
                    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i, 0]]
                    frame_with_inliers = cv2.drawMatches(
                        marker_image, keypoints_marker, frame, keypoints_frame,
                        inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

                    # Record results
                    processing_time = (time.time() - start_time) * 1000  # Convert to ms
                    results.append({
                        "Frame": len(results) + 1,
                        "Total Matches": len(matches),
                        "Good Matches": len(good_matches),
                        "Inliers": np.sum(mask),
                        "Inlier Ratio": inlier_ratio,
                        "Reprojection Error": reprojection_error,
                        "Processing Time (ms)": processing_time
                    })

                    # Display the combined frame
                    combined_frame = np.hstack((frame_with_overlay, frame_with_inliers))
                    cv2.imshow('Overlay and Inliers Side by Side', combined_frame)

                    print(f"Inlier Ratio: {inlier_ratio:.2f}, Reprojection Error: {reprojection_error:.2f}, Time: {processing_time:.2f} ms")

                else:
                    print("Homography could not be computed.")
            else:
                cv2.putText(frame, "Not enough matches", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Descriptor Matching', frame)
        else:
            cv2.imshow('Descriptor Matching', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Display the table of results
    print("\nAnalysis Results:")
    print(df)

    # Save results to a CSV (optional)
    df.to_csv("results.csv", index=False)

    # Display keypoint matches for the last frame
    if len(results) > 0:
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(frame_with_inliers, cv2.COLOR_BGR2RGB))
        plt.title("Keypoint Matches - Last Frame")
        plt.axis("off")
        plt.show()