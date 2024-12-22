Here's the corrected README content for your project with the two images displayed side by side:

---

# Parking Spot Detection Project

## Overview

This parking spot detection project consists of a dataset folder, `main.py`, `util.py`, `requirements.txt`, and the trained model stored in `model.pkl`. Below is a breakdown of the project components and an explanation of the code and its functionality.

## Project Structure

### Dataset Folder

The dataset folder includes the following:

1. **Data**: Contains video footage (`parking_1920_1080_loop.mp4`) used for parking spot detection and a mask image (`mask_1920_1080.png`) that identifies parking spots in the frame.
2. **Graphs**: Includes visual results or performance metrics obtained from the trained classifier. These could be histograms, accuracy plots, or confusion matrices.

### Code Explanation

#### `util.py`

This script contains utility functions essential for parking spot detection. The core functions are:

1. **`empty_or_not(spot_bgr)`**

   This function determines whether a parking spot is empty or occupied using the trained SVM model stored in `model.pkl`.

   - The image of the parking spot (`spot_bgr`) is resized to 15x15 pixels.
   - The resized image is flattened into a one-dimensional array and passed to the trained model for prediction.
   - The function returns `True` for "empty" and `False` for "not_empty" based on the model's output.

2. **`get_parking_spots_bboxes(connected_components)`**

   Extracts bounding box coordinates for all parking spots from the connected components analysis of the mask image.

   - It iterates over the labeled regions from the mask image.
   - For each region, it calculates the coordinates (top-left corner, width, height).
   - The bounding box coordinates for all spots are returned as a list.

#### `main.py`

This script is the primary entry point for the parking spot detection system. It handles video processing, spot occupancy detection, and real-time visualization.

1. **Mask and Video Input**

   The script loads the mask image and video file paths. The mask identifies parking spots, and the video simulates a parking lot.

2. **Connected Components Analysis**

   It uses `cv2.connectedComponentsWithStats` to extract parking spot bounding boxes from the mask.

3. **Video Frame Processing**

   - Each video frame is processed at a defined interval (`step`).
   - Differences between the current and previous frames are calculated to determine movement or changes in the spots.
   - For each parking spot, the `empty_or_not` function is called to classify the spot as empty or occupied.

4. **Visualization**

   - The status of each spot is displayed in the video frame using colored rectangles (green for empty, red for occupied).
   - The number of available spots is also shown as text on the frame.

### `requirements.txt`

This file lists the dependencies for the project, including libraries like `opencv-python`, `numpy`, `matplotlib`, and `scikit-image`.

### Usage Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Main Script**:
   ```bash
   python main.py
   ```

### Visual Results

Here you can include images or screenshots showcasing the results of the parking spot detection system:

<p align="center">
  <img src="parking-spot-detection
/mask_1920_1080.png
" alt="Initial Frame" width="45%">
  <img src="parking-spot-detection
/Screenshot 2024-12-20 170157.png" alt="Detected Spots" width="45%">
</p>

### Contributing

Feel free to fork this project, make improvements, and submit pull requests. Contributions are welcome!

### License

This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact

For any questions or inquiries, please contact the project maintainer at [your-email@example.com].

---

Replace the placeholders with the actual paths to your images. This README file now includes sections where you can input image results, making it easy to showcase the visual outcomes of your project.
