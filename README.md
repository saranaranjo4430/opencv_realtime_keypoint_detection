# Marker Replacement and Augmented Reality Pipeline

This repository contains the implementation of a real-time marker replacement pipeline using OpenCV. The project showcases keypoint detection, descriptor matching, homography estimation, and marker overlay on live video. It includes two main files:

## **Files**

1. **`Final_Project_Sara_Naranjo.ipynb`**
   - This Jupyter Notebook provides a detailed explanation of the project, step by step.
   - It includes keypoint detection, descriptor visualization, inlier ratio analysis, and marker replacement.
   - Due to the limitations of Jupyter Notebook, real-time detection is not implemented. Instead, Matplotlib is used to display the camera feed as individual images, updated dynamically.

2. **`PythonFile.py`**
   - This standalone Python script implements the full pipeline with **real-time detection**.
   - It uses `cv2.imshow()` to display the live video feed with the marker overlay, offering real-time results.

## **Usage**

- For detailed project steps and analysis, open and run `ProjectNotebook.ipynb` in Jupyter Notebook.
- For a real-time demonstration of the marker replacement pipeline, execute `PythonFile.py` in a terminal or Python IDE.
