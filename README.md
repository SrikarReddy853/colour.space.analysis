#lab-1
# color_space_analysis.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# ----- CONFIG -----
IMG_PATH = "/content/wp3.jpg"   # path to the cat image you gave
OUT_DIR = "/mnt/data/color_analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----- HELPERS -----
def print_image_info(path, img):
    # file info
    filesize = os.path.getsize(path)
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    dtype = img.dtype
    bit_depth = img.dtype.itemsize * 8  # bits per channel
    print(f"File: {path}")
    print(f" - Width x Height: {w} x {h}")
    print(f" - Channels (array): {channels}")
    print(f" - Dtype: {dtype} -> {bit_depth} bits per channel")
    print(f" - File size: {filesize/1024:.2f} KB")

def show_grid(images, titles, figsize=(12,8), cmap='gray', save_as=None):
    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=figsize)
    for i, (im, t) in enumerate(zip(images, titles), 1):
        plt.subplot(rows, cols, i)
        # if image has 3 channels and dtype isn't float normalized, convert BGR->RGB for display
        if im is None:
            plt.title(t)
            plt.axis('off')
            continue
        if im.ndim == 3:
            # assume BGR (OpenCV) — convert to RGB for matplotlib display
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        else:
            plt.imshow(im, cmap=cmap)
            plt.axis('off')
        plt.title(t)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
        print(f"Saved visualization -> {save_as}")
    plt.show()

def normalize_for_display(channel):
    """ Normalize single-channel image to 0-255 uint8 for display """
    ch = channel.astype(np.float32)
    ch_min, ch_max = ch.min(), ch.max()
    if ch_max - ch_min < 1e-6:
        return np.zeros_like(ch, dtype=np.uint8)
    norm = 255 * (ch - ch_min) / (ch_max - ch_min)
    return norm.astype(np.uint8)

def color_highlight_channel(bgr, channel_index):
    """
    Return an image that highlights one channel in its color while zeroing others.
    channel_index: 0=B,1=G,2=R
    """
    zeros = np.zeros_like(bgr[:,:,0])
    ch = bgr[:,:,channel_index]
    if channel_index == 2:   # R highlighted: produce (0,0,R) then convert to BGR format - but OpenCV uses BGR so merge order is B,G,R
        merged = cv2.merge([zeros, zeros, ch])
    elif channel_index == 1:
        merged = cv2.merge([zeros, ch, zeros])
    else:
        merged = cv2.merge([ch, zeros, zeros])
    return merged

# ----- MAIN PROCESS -----
# 1) Load image (preserve channels)
img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image at {IMG_PATH}")

print_image_info(IMG_PATH, img_bgr)

# If image has alpha channel, drop it (common for PNG). We'll note if alpha existed.
if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
    print(" - Note: Image has alpha channel; dropping alpha for color-space conversions.")
    img_bgr = img_bgr[:, :, :3]

# 2) Basic RGB (OpenCV returns BGR)
b, g, r = cv2.split(img_bgr)

# 3) Convert to HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)
# OpenCV Hue range: [0,179], Saturation and Value: [0,255]

# 4) Convert to YCbCr (OpenCV uses YCrCb ordering)
# OpenCV color conversion constant: COLOR_BGR2YCrCb returns channels in order [Y, Cr, Cb]
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
Y = img_ycrcb[:, :, 0]
Cr = img_ycrcb[:, :, 1]
Cb = img_ycrcb[:, :, 2]
# If you want the more commonly named order Y, Cb, Cr, we can swap:
# Y, Cb, Cr = img_ycrcb[..., 0], img_ycrcb[..., 2], img_ycrcb[..., 1]

# 5) Visualize channels
# Prepare grayscale visualizations (normalized)
r_gray = normalize_for_display(r)
g_gray = normalize_for_display(g)
b_gray = normalize_for_display(b)

h_display = normalize_for_display(h)        # Hue normalized for visualization
s_display = normalize_for_display(s)
v_display = normalize_for_display(v)

Y_gray = normalize_for_display(Y)
Cb_gray = normalize_for_display(Cb)
Cr_gray = normalize_for_display(Cr)

# Color-highlighted R/G/B (so human sees the color contribution)
r_color = color_highlight_channel(img_bgr, 2)
g_color = color_highlight_channel(img_bgr, 1)
b_color = color_highlight_channel(img_bgr, 0)

# Show a grid: original, R_color, G_color, B_color, H, S, V, Y, Cb, Cr
images = [
    img_bgr,
    r_color, g_color, b_color,
    h_display, s_display, v_display,
    Y_gray, Cb_gray, Cr_gray
]
titles = [
    "Original (BGR)",
    "Red channel (highlighted)", "Green channel (highlighted)", "Blue channel (highlighted)",
    "Hue (normalized)", "Saturation", "Value (luminance proxy)",
    "Y (luma)", "Cb (blue diff)", "Cr (red diff)"
]
show_grid(images, titles, figsize=(14,10), save_as=os.path.join(OUT_DIR, "all_channels.png"))

# Save individual channel images (optional)
cv2.imwrite(os.path.join(OUT_DIR, "orig_rgb.png"), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
cv2.imwrite(os.path.join(OUT_DIR, "r_gray.png"), r_gray)
cv2.imwrite(os.path.join(OUT_DIR, "g_gray.png"), g_gray)
cv2.imwrite(os.path.join(OUT_DIR, "b_gray.png"), b_gray)
cv2.imwrite(os.path.join(OUT_DIR, "h.png"), h_display)
cv2.imwrite(os.path.join(OUT_DIR, "s.png"), s_display)
cv2.imwrite(os.path.join(OUT_DIR, "v.png"), v_display)
cv2.imwrite(os.path.join(OUT_DIR, "Y.png"), Y_gray)
cv2.imwrite(os.path.join(OUT_DIR, "Cb.png"), Cb_gray)
cv2.imwrite(os.path.join(OUT_DIR, "Cr.png"), Cr_gray)

print("Saved per-channel images to:", OUT_DIR)

# 6) Analysis: dominant hue + luminance stats
# Build a histogram of hue values
h_flat = h.flatten()
# Hue values in OpenCV range [0,179]. Count most frequent value (mode)
counts = Counter(h_flat.tolist())
most_common_hue, count = counts.most_common(1)[0]
total = len(h_flat)
percentage = 100.0 * count / total

# Convert hue to degrees: OpenCV H*2 => degrees in [0,360)
hue_degrees = most_common_hue * 2

print(f"\nDominant hue (mode): {most_common_hue} (OpenCV units) ≈ {hue_degrees:.1f}° ; frequency {count}/{total} = {percentage:.2f}%")
print("Saturation mean:", float(np.mean(s)))
print("Value (V) mean (luminance proxy):", float(np.mean(v)))
print("Y (luma) mean:", float(np.mean(Y)))
print("Y (luma) std dev:", float(np.std(Y)))

# 7) Quick interpretation (text output)
print("\nQuick interpretation:")
print(" - Hue (H) shows dominant color (here printed as a hue angle).")
print(" - S (saturation) indicates color purity (low = more gray).")
print(" - V (value) and Y channel indicate brightness/luminance.")
print(" - Cb/Cr are chroma components; useful when separating color from intensity (tracking, compression).")
print("\nSituations where each color model is useful:")
print(" - RGB/BGR: low-level pixel manipulation, direct display, simple tasks.")
print(" - HSV: color-based segmentation or tracking (select hue ranges regardless of lightness).")
print(" - YCbCr: video/compression and tasks that separate luminance from chrominance (useful for compression, skin detection with chroma thresholds).")

# 8) Optionally show a hue histogram plot
plt.figure(figsize=(8,3))
plt.hist(h_flat, bins=180, range=(0,180))
plt.title("Hue histogram (OpenCV 0-179)")
plt.xlabel("Hue (OpenCV units)")
plt.ylabel("Count")
plt.tight_layout()
hist_path = os.path.join(OUT_DIR, "hue_histogram.png")
plt.savefig(hist_path)
print("Saved hue histogram ->", hist_path)
plt.show()
