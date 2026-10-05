import os
import cv2
import numpy as np

# --------------------------------------------------
# STEP 1: Load image
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check local folder first, then Downloads
candidate_paths = [
    os.path.join(SCRIPT_DIR, "Crushed_ia.jpeg"),
    r"C:\Users\Hp\Downloads\Crushed_ia.jpeg",
]

IMAGE_PATH = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
OUTPUT_DIR = SCRIPT_DIR

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

# --------------------------------------------------
# STEP 2: Convert RGB image to grayscale
# --------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# STEP 3: Otsu automatic thresholding
# --------------------------------------------------
threshold, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# --------------------------------------------------
# STEP 4: Count pixels
# --------------------------------------------------
total_pixels = binary.size
silver_pixels = np.count_nonzero(binary == 255)
black_pixels = np.count_nonzero(binary == 0)

# --------------------------------------------------
# STEP 5: Calculate percentages
# --------------------------------------------------
silver_percentage = (silver_pixels / total_pixels) * 100
black_percentage = (black_pixels / total_pixels) * 100

# --------------------------------------------------
# STEP 6: Display results in terminal
# --------------------------------------------------
print("==================================================")
print("             CRUSHED IA IMAGE ANALYSIS            ")
print("==================================================")
print(f"Image Path       : {IMAGE_PATH}")
print(f"Image dimensions : {gray.shape[1]} x {gray.shape[0]} (W x H)")
print(f"Total pixels     : {total_pixels:,}")
print(f"Otsu threshold   : {threshold:.2f}")
print(f"Silver/White px  : {silver_pixels:,} ({silver_percentage:.2f}%)")
print(f"Black/Void px    : {black_pixels:,} ({black_percentage:.2f}%)")
print("==================================================")

# --------------------------------------------------
# STEP 7: Save binary segmentation mask
# --------------------------------------------------
seg_output_path = os.path.join(OUTPUT_DIR, "IA_segmentation_result.png")
cv2.imwrite(seg_output_path, binary)
print(f"[+] Segmentation image saved to: {seg_output_path}")

# --------------------------------------------------
# STEP 8: Create and save visual report with text overlay
# --------------------------------------------------
report = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# Background banner card for legible text
overlay_h, overlay_w = 240, min(480, report.shape[1] - 20)
cv2.rectangle(report, (10, 10), (10 + overlay_w, 10 + overlay_h), (30, 30, 30), -1)
cv2.rectangle(report, (10, 10), (10 + overlay_w, 10 + overlay_h), (0, 255, 255), 2)

font = cv2.FONT_HERSHEY_SIMPLEX
text_lines = [
    "IA PIXEL AREA ANALYSIS",
    f"Otsu Threshold : {threshold:.1f}",
    f"Total Pixels   : {total_pixels:,}",
    f"Silver Material: {silver_pixels:,}",
    f"Void/Black Area: {black_pixels:,}",
    f"Material Ratio : {silver_percentage:.2f}%",
    f"Void Ratio     : {black_percentage:.2f}%"
]

y = 40
for idx, line in enumerate(text_lines):
    color = (0, 255, 255) if idx == 0 else (255, 255, 255)
    scale = 0.65 if idx == 0 else 0.55
    thickness = 2 if idx == 0 else 1
    cv2.putText(report, line, (25, y), font, scale, color, thickness, cv2.LINE_AA)
    y += 28

report_output_path = os.path.join(OUTPUT_DIR, "IA_area_analysis_report.png")
cv2.imwrite(report_output_path, report)
print(f"[+] Visual report saved to: {report_output_path}")

# --------------------------------------------------
# STEP 9: Optional interactive display
# --------------------------------------------------
# Run with --show flag to display interactive popup windows
import sys
if "--show" in sys.argv or os.environ.get("SHOW_PREVIEW", "0") == "1":
    print("\n[i] Displaying popup windows. Press any key on an image window to exit...")
    cv2.imshow("Original Image", image)
    cv2.imshow("Binary Segmentation", binary)
    cv2.imshow("Analysis Report", report)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("\n[Tip] Run with '--show' (e.g. `python crushed_ia.py --show`) to display interactive preview windows.")

