"""
Pupil Detection & Facial Feature Localization
Detects pupil positions, facial features, and calculates inter-pupil distance.
Supports non-frontal faces: tilted, side-view, looking down, etc.
"""

import sys
import math
import cv2
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ToolTracker:
    """Records the order of tools used in each run."""

    def __init__(self):
        self.steps = []

    def log(self, tool_name):
        self.steps.append(tool_name)

    def report(self):
        if self.steps:
            print(f"Tool order: {' -> '.join(self.steps)}")
        else:
            print("No tools used")


# ---------------------------------------------------------------------------
# Image processing tools
# ---------------------------------------------------------------------------

def gaussian_blur(img, ksize=7, tracker=None):
    """Gaussian Blur"""
    if tracker:
        tracker.log("Gaussian Blur")
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def binarization(img, thresh=50, max_val=255, tracker=None, use_otsu=False):
    """Binarization"""
    if tracker:
        tracker.log("Binarization")
    if use_otsu:
        _, binary = cv2.threshold(img, 0, max_val,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, thresh, max_val, cv2.THRESH_BINARY_INV)
    return binary


def sobel_edge(img, tracker=None):
    """Sobel"""
    if tracker:
        tracker.log("Sobel")
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(np.clip(magnitude, 0, 255))


def canny_edge(img, low=30, high=100, tracker=None):
    """Canny"""
    if tracker:
        tracker.log("Canny")
    return cv2.Canny(img, low, high)


def find_contours(binary_img, tracker=None):
    """Contour"""
    if tracker:
        tracker.log("Contour")
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def connected_components(binary_img, tracker=None):
    """Connected Component Labeling"""
    if tracker:
        tracker.log("Connected Component Labeling")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    return num_labels, labels, stats, centroids


def hough_circles(img, dp=1.2, min_dist=30, param1=100, param2=20,
                  min_radius=3, max_radius=80, tracker=None):
    """Hough"""
    if tracker:
        tracker.log("Hough")
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    return circles


def perspective_transform(img, src_pts, dst_pts, out_size, tracker=None):
    """Perspective Transform"""
    if tracker:
        tracker.log("Perspective Transform")
    M = cv2.getPerspectiveTransform(
        np.float32(src_pts), np.float32(dst_pts)
    )
    return cv2.warpPerspective(img, M, out_size)


def reference_pt(landmarks, indices, tracker=None):
    """Reference Pt"""
    if tracker:
        tracker.log("Reference Pt")
    return [(landmarks[i].x, landmarks[i].y) for i in indices]


class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# ---------------------------------------------------------------------------
# Multi-angle face detection
# ---------------------------------------------------------------------------

def detect_face_multi_angle(gray):
    """Try multiple cascades + rotation angles to find faces.
    Returns (face_rect, angle, rotated_gray) or None.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    angles = [0, -15, 15, -30, 30, -45, 45]

    for angle in angles:
        if angle == 0:
            rotated = gray
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h))

        for sf, mn in [(1.1, 5), (1.05, 3), (1.1, 3)]:
            faces = face_cascade.detectMultiScale(
                rotated, scaleFactor=sf, minNeighbors=mn, minSize=(30, 30)
            )
            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                return face, angle, rotated

        for sf, mn in [(1.1, 3), (1.05, 3)]:
            faces = profile_cascade.detectMultiScale(
                rotated, scaleFactor=sf, minNeighbors=mn, minSize=(30, 30)
            )
            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                return face, angle, rotated

            flipped = cv2.flip(rotated, 1)
            faces = profile_cascade.detectMultiScale(
                flipped, scaleFactor=sf, minNeighbors=mn, minSize=(30, 30)
            )
            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = face
                face = (w - fx - fw, fy, fw, fh)
                return face, angle, rotated

    return None


# ---------------------------------------------------------------------------
# Eye detection within face ROI using cascade
# ---------------------------------------------------------------------------

def detect_eyes_in_face(face_roi_gray):
    """Detect eyes within a face ROI using multiple cascades.
    Returns list of (ex, ey, ew, eh) in face ROI coordinates.
    """
    all_eyes = []
    for cascade_name in ["haarcascade_eye.xml",
                         "haarcascade_lefteye_2splits.xml",
                         "haarcascade_righteye_2splits.xml",
                         "haarcascade_eye_tree_eyeglasses.xml"]:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + cascade_name
        )
        if cascade.empty():
            continue
        eyes = cascade.detectMultiScale(face_roi_gray, scaleFactor=1.05,
                                        minNeighbors=3, minSize=(10, 10))
        if len(eyes) > 0:
            all_eyes.extend(eyes.tolist())

    if not all_eyes:
        return []

    # Merge overlapping detections
    all_eyes = sorted(all_eyes, key=lambda e: e[2] * e[3], reverse=True)
    merged = [all_eyes[0]]
    for e in all_eyes[1:]:
        overlap = False
        for m in merged:
            if (abs(e[0] - m[0]) < m[2] * 0.5 and
                    abs(e[1] - m[1]) < m[3] * 0.5):
                overlap = True
                break
        if not overlap:
            merged.append(e)

    # Filter: eyes should be in the upper 45% of the face
    fh = face_roi_gray.shape[0]
    fw = face_roi_gray.shape[1]
    merged = [e for e in merged if e[1] + e[3] // 2 < fh * 0.45]

    # Filter: eye width should be reasonable (10%-50% of face width)
    merged = [e for e in merged if fw * 0.08 < e[2] < fw * 0.55]

    # If 2+ eyes found, verify they are roughly at the same height
    if len(merged) >= 2:
        merged = sorted(merged, key=lambda e: e[0])
        heights = [e[1] + e[3] // 2 for e in merged]
        # Keep pairs where y-difference < 30% of face height
        valid = [merged[0]]
        for e in merged[1:]:
            y_diff = abs((e[1] + e[3] // 2) - heights[0])
            if y_diff < fh * 0.15:
                valid.append(e)
        merged = valid

    # Sort by x (left to right), take top 2
    merged = sorted(merged, key=lambda e: e[0])[:2]
    return merged


# ---------------------------------------------------------------------------
# Tilt correction
# ---------------------------------------------------------------------------

def correct_eye_tilt(gray, face_rect, tracker):
    """Detect eye tilt and correct using Perspective Transform."""
    fx, fy, fw, fh = face_rect
    face_roi = gray[fy:fy + fh, fx:fx + fw]

    eyes = detect_eyes_in_face(face_roi)
    if len(eyes) < 2:
        return gray, face_rect, []

    left_eye_center = (fx + eyes[0][0] + eyes[0][2] // 2,
                       fy + eyes[0][1] + eyes[0][3] // 2)
    right_eye_center = (fx + eyes[1][0] + eyes[1][2] // 2,
                        fy + eyes[1][1] + eyes[1][3] // 2)

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = math.degrees(math.atan2(dy, dx))

    if abs(angle) < 2.0:
        return gray, face_rect, eyes

    if abs(angle) > 35.0:
        print(f"Eye tilt {angle:.1f} deg is abnormal (>35 deg), skipping correction")
        return gray, face_rect, eyes

    print(f"Face tilted {angle:.1f} deg, applying Perspective Transform...")

    img_h, img_w = gray.shape[:2]
    eye_mid = ((left_eye_center[0] + right_eye_center[0]) / 2,
               (left_eye_center[1] + right_eye_center[1]) / 2)

    cos_a = math.cos(math.radians(-angle))
    sin_a = math.sin(math.radians(-angle))

    def rotate_point(px, py, cx, cy):
        ddx, ddy = px - cx, py - cy
        return (cx + ddx * cos_a - ddy * sin_a, cy + ddx * sin_a + ddy * cos_a)

    corners = [(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)]
    src_pts = [list(c) for c in corners]
    dst_pts = [list(rotate_point(c[0], c[1], eye_mid[0], eye_mid[1]))
               for c in corners]

    corrected = perspective_transform(gray, src_pts, dst_pts, (img_w, img_h),
                                      tracker=tracker)

    # Re-detect eyes in corrected image
    face_roi_new = corrected[fy:fy + fh, fx:fx + fw]
    new_eyes = detect_eyes_in_face(face_roi_new)
    if len(new_eyes) >= 2:
        eyes = new_eyes

    return corrected, face_rect, eyes


# ---------------------------------------------------------------------------
# Pupil detection pipeline
# ---------------------------------------------------------------------------

def detect_pupil_in_roi(roi, tracker, min_pupil_r=4):
    """Detect pupil in an eye ROI.
    Returns (cx, cy, radius) in ROI coordinates or None.
    """
    h, w = roi.shape[:2]
    if h < 10 or w < 10:
        return None

    # Crop to central region (remove eyelid corners that confuse detection)
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.15)
    inner = roi[margin_y:h - margin_y, margin_x:w - margin_x]
    ih, iw = inner.shape[:2]
    if ih < 8 or iw < 8:
        inner = roi
        margin_x, margin_y = 0, 0
        ih, iw = h, w

    # Max pupil radius: scale by inner ROI size
    # For small ROIs (<40px), allow up to 40%; for larger, 25%
    if min(iw, ih) < 40:
        max_pupil_r = max(min_pupil_r + 2, int(min(iw, ih) * 0.45))
    else:
        max_pupil_r = max(min_pupil_r + 2, int(min(iw, ih) * 0.30))

    # 1) Strong Gaussian Blur
    blurred = gaussian_blur(inner, ksize=7, tracker=tracker)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

    # 2) OTSU Binarization
    binary = binarization(blurred, tracker=tracker, use_otsu=True)

    # 3) Erode to shrink from iris boundary to pupil core
    erode_k = max(3, int(min(iw, ih) * 0.08))
    if erode_k % 2 == 0:
        erode_k += 1
    erode_iter = 1 if min(iw, ih) < 40 else 2
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    binary = cv2.erode(binary, erode_kernel, iterations=erode_iter)

    # 4) Morphology: close gaps, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 5) Connected Component Labeling to find dark blobs
    num_labels, labels, stats, centroids = connected_components(binary, tracker=tracker)

    ccl_best = None
    ccl_best_score = -1
    center_x, center_y = iw / 2, ih / 2
    max_dist = math.hypot(center_x, center_y)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cx_c, cy_c = centroids[i]

        r_est = int((bw + bh) / 4)
        if r_est < min_pupil_r or r_est > max_pupil_r:
            continue
        if area < min_pupil_r * min_pupil_r:
            continue

        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

        dist = math.hypot(cx_c - center_x, cy_c - center_y)
        pos_score = 1 - (dist / max_dist) if max_dist > 0 else 0

        mask = (labels == i).astype(np.uint8) * 255
        mean_val = cv2.mean(inner, mask=mask)[0]
        dark_score = 1 - (mean_val / 255.0)

        score = aspect * 0.20 + pos_score * 0.25 + dark_score * 0.55
        if score > ccl_best_score:
            ccl_best_score = score
            ccl_best = (int(cx_c) + margin_x, int(cy_c) + margin_y, r_est)

    # 6) Contour-based detection
    edges = canny_edge(blurred, low=30, high=80, tracker=tracker)
    combined = cv2.bitwise_or(binary, edges)
    contours = find_contours(combined, tracker=tracker)

    cnt_best = None
    cnt_best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_pupil_r * min_pupil_r:
            continue
        ((cx_c, cy_c), radius) = cv2.minEnclosingCircle(cnt)
        if radius < min_pupil_r or radius > max_pupil_r:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)

        dist = math.hypot(cx_c - center_x, cy_c - center_y)
        pos_score = 1 - (dist / max_dist) if max_dist > 0 else 0

        c_mask = np.zeros((ih, iw), dtype=np.uint8)
        cv2.drawContours(c_mask, [cnt], 0, 255, -1)
        mean_val = cv2.mean(inner, mask=c_mask)[0]
        dark_score = 1 - (mean_val / 255.0)

        score = circularity * 0.20 + pos_score * 0.25 + dark_score * 0.55
        if score > cnt_best_score:
            cnt_best_score = score
            cnt_best = (int(cx_c) + margin_x, int(cy_c) + margin_y,
                        max(int(radius), min_pupil_r))

    # 7) Hough circle detection as fallback
    hough_min_r = max(min_pupil_r, int(min(iw, ih) * 0.06))
    hough_max_r = max(hough_min_r + 1, max_pupil_r)
    circles = hough_circles(
        blurred, dp=1.5, min_dist=max(iw // 3, 10),
        param1=80, param2=15,
        min_radius=hough_min_r, max_radius=hough_max_r,
        tracker=tracker,
    )

    hough_best = None
    if circles is not None:
        hough_best_score = -1
        for c in circles[0]:
            cx_c, cy_c, r = c
            dist = math.hypot(cx_c - center_x, cy_c - center_y)
            pos_s = 1 - (dist / max_dist) if max_dist > 0 else 0
            h_mask = np.zeros((ih, iw), dtype=np.uint8)
            cv2.circle(h_mask, (int(cx_c), int(cy_c)), max(int(r), 1), 255, -1)
            dark_s = 1 - (cv2.mean(inner, mask=h_mask)[0] / 255.0)
            h_score = pos_s * 0.3 + dark_s * 0.7
            if h_score > hough_best_score:
                hough_best_score = h_score
                hough_best = (int(cx_c) + margin_x, int(cy_c) + margin_y,
                              max(int(r), min_pupil_r))

    # Pick best result: prioritize darkness
    candidates = []
    if ccl_best:
        candidates.append(("CCL", ccl_best, ccl_best_score))
    if cnt_best:
        candidates.append(("Contour", cnt_best, cnt_best_score))
    if hough_best:
        candidates.append(("Hough", hough_best, hough_best_score if circles is not None else 0))

    if not candidates:
        return None

    # Choose the candidate with the best score
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Facial feature detection
# ---------------------------------------------------------------------------

def detect_facial_features(gray, face_rect, eyes_in_face, output, tracker):
    """Detect facial features and draw on output image.
    eyes_in_face: list of (ex, ey, ew, eh) in face ROI coordinates.
    """
    fx, fy, fw, fh = face_rect
    face_roi_gray = gray[fy:fy + fh, fx:fx + fw]
    features = {}

    # --- Eyes (trim top 30% of cascade box to exclude eyebrows) ---
    if len(eyes_in_face) >= 2:
        for i, (ex, ey, ew, eh) in enumerate(eyes_in_face[:2]):
            label = "L-Eye" if i == 0 else "R-Eye"
            # Trim: keep only lower 65% of cascade box (eye area only)
            trim_top = int(eh * 0.35)
            ey_trimmed = ey + trim_top
            eh_trimmed = eh - trim_top
            center = (fx + ex + ew // 2, fy + ey_trimmed + eh_trimmed // 2)
            features[label] = center
            cv2.rectangle(output, (fx + ex, fy + ey_trimmed),
                          (fx + ex + ew, fy + ey_trimmed + eh_trimmed),
                          (255, 255, 0), 1)
            cv2.putText(output, label, (fx + ex, fy + ey_trimmed - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # --- Nose detection ---
    # Strategy: use Gaussian Blur + Sobel + Canny + Contour on nose-tip region
    # Nose tip is at ~55-75% of face height, center 40% width
    nose_detected = False
    nose_marker_r = max(8, int(fw * 0.03))  # visible marker size

    # Try cascade first
    nose_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_mcs_nose.xml"
    )
    if not nose_cascade.empty():
        nose_ry = int(fh * 0.30)
        nose_rh = int(fh * 0.45)
        nose_roi = face_roi_gray[nose_ry:nose_ry + nose_rh, :]
        if nose_roi.shape[0] > 5 and nose_roi.shape[1] > 5:
            noses = nose_cascade.detectMultiScale(nose_roi, 1.1, 3, minSize=(10, 10))
            if len(noses) > 0:
                nx, ny, nw, nh = max(noses, key=lambda n: n[2] * n[3])
                # Nose tip = bottom center of cascade box
                nc = (fx + nx + nw // 2, fy + nose_ry + ny + int(nh * 0.75))
                features["Nose"] = nc
                cv2.rectangle(output, (fx + nx, fy + nose_ry + ny),
                              (fx + nx + nw, fy + nose_ry + ny + nh),
                              (0, 255, 255), 1)
                cv2.circle(output, nc, nose_marker_r, (0, 255, 255), 2)
                cv2.putText(output, "Nose", (nc[0] - 20, nc[1] - nose_marker_r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                nose_detected = True

    if not nose_detected:
        # Fallback: multi-pass Sobel + Canny on nose-tip region (55-75% height)
        nose_top = int(fh * 0.50)
        nose_bot = int(fh * 0.75)
        nose_left = int(fw * 0.30)
        nose_right = int(fw * 0.70)
        na = face_roi_gray[nose_top:nose_bot, nose_left:nose_right]
        if na.shape[0] > 5 and na.shape[1] > 5:
            # Pass 1: Gaussian Blur
            na_blur = gaussian_blur(na, ksize=7, tracker=tracker)
            # Pass 2: Sobel to find nose ridge edges
            na_sobel = sobel_edge(na_blur, tracker=tracker)
            # Pass 3: Gaussian Blur again on Sobel result
            na_sobel = gaussian_blur(na_sobel, ksize=5, tracker=tracker)
            # Pass 4: Canny for sharper edges
            na_canny = canny_edge(na_blur, low=30, high=80, tracker=tracker)
            # Combine
            na_combined = cv2.bitwise_or(na_sobel, na_canny)
            # Pass 5: Binarization
            na_bin = binarization(na_combined, thresh=40, tracker=tracker)
            # Pass 6: Contour
            na_cnts = find_contours(na_bin, tracker=tracker)

            if na_cnts:
                # Find the most central contour (nose tip)
                na_h, na_w = na.shape[:2]
                na_cx, na_cy = na_w / 2, na_h / 2
                best_cnt = None
                best_score = -1
                for cnt in na_cnts:
                    area = cv2.contourArea(cnt)
                    if area < 20:
                        continue
                    M_c = cv2.moments(cnt)
                    if M_c["m00"] == 0:
                        continue
                    cx_c = M_c["m10"] / M_c["m00"]
                    cy_c = M_c["m01"] / M_c["m00"]
                    dist = math.hypot(cx_c - na_cx, cy_c - na_cy)
                    max_d = math.hypot(na_cx, na_cy)
                    pos = 1 - (dist / max_d) if max_d > 0 else 0
                    score = pos * 0.6 + (area / (na_w * na_h)) * 0.4
                    if score > best_score:
                        best_score = score
                        best_cnt = cnt

                if best_cnt is not None:
                    M_c = cv2.moments(best_cnt)
                    ncx = int(M_c["m10"] / M_c["m00"])
                    ncy = int(M_c["m01"] / M_c["m00"])
                    nc = (fx + nose_left + ncx, fy + nose_top + ncy)
                    features["Nose"] = nc
                    cv2.circle(output, nc, nose_marker_r, (0, 255, 255), 2)
                    cv2.putText(output, "Nose",
                                (nc[0] - 20, nc[1] - nose_marker_r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    nose_detected = True

    # 偵測不到就不標記，不硬猜位置

    # --- Mouth (Canny + Contour in lower face) ---
    mouth_detected = False
    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_mcs_mouth.xml"
    )
    if not mouth_cascade.empty():
        mry = int(fh * 0.55)
        mrh = fh - mry
        mouth_roi = face_roi_gray[mry:mry + mrh, :]
        if mouth_roi.shape[0] > 5 and mouth_roi.shape[1] > 5:
            mouths = mouth_cascade.detectMultiScale(mouth_roi, 1.1, 5, minSize=(15, 10))
            if len(mouths) > 0:
                mx, my, mw, mh = max(mouths, key=lambda m: m[2] * m[3])
                mc = (fx + mx + mw // 2, fy + mry + my + mh // 2)
                features["Mouth"] = mc
                cv2.rectangle(output, (fx + mx, fy + mry + my),
                              (fx + mx + mw, fy + mry + my + mh), (255, 0, 255), 1)
                cv2.putText(output, "Mouth", (fx + mx, fy + mry + my - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                mouth_detected = True

    if not mouth_detected:
        mry = int(fh * 0.60)
        mrh = fh - mry
        mouth_roi = face_roi_gray[mry:mry + mrh, :]
        if mouth_roi.shape[0] > 5 and mouth_roi.shape[1] > 5:
            mb = gaussian_blur(mouth_roi, ksize=5, tracker=tracker)
            me = canny_edge(mb, low=50, high=120, tracker=tracker)
            mc_list = find_contours(me, tracker=tracker)
            best_m = None
            best_w = 0
            for cnt in mc_list:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw / max(bh, 1) > 1.5 and bw > fw * 0.15 and bw > best_w:
                    best_w = bw
                    best_m = (bx, by, bw, bh)
            if best_m:
                bx, by, bw, bh = best_m
                mc = (fx + bx + bw // 2, fy + mry + by + bh // 2)
                features["Mouth"] = mc
                cv2.rectangle(output, (fx + bx, fy + mry + by),
                              (fx + bx + bw, fy + mry + by + bh), (255, 0, 255), 1)
                cv2.putText(output, "Mouth", (fx + bx, fy + mry + by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                mouth_detected = True

    # 嘴巴偵測不到就不標記

    # --- Eyebrows (Sobel + Contour above eyes) ---
    eb_region = face_roi_gray[int(fh * 0.05):int(fh * 0.30),
                              int(fw * 0.05):int(fw * 0.95)]
    if eb_region.shape[0] > 5 and eb_region.shape[1] > 5:
        eb_b = gaussian_blur(eb_region, ksize=5, tracker=tracker)
        eb_s = sobel_edge(eb_b, tracker=tracker)
        eb_bin = binarization(eb_s, thresh=40, tracker=tracker)
        eb_cnts = find_contours(eb_bin, tracker=tracker)

        eb_cands = []
        for cnt in eb_cnts:
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            if w_c / max(h_c, 1) > 2.0 and cv2.contourArea(cnt) > 30:
                eb_cands.append((x_c, y_c, w_c, h_c))

        if eb_cands:
            eb_cands.sort(key=lambda c: c[0])
            ox = int(fw * 0.05)
            oy = int(fh * 0.05)
            for i, (bx, by, bw, bh) in enumerate(eb_cands[:2]):
                label = "L-Brow" if bx < eb_region.shape[1] // 2 else "R-Brow"
                ec = (fx + ox + bx + bw // 2, fy + oy + by + bh // 2)
                features[label] = ec
                cv2.rectangle(output,
                              (fx + ox + bx, fy + oy + by),
                              (fx + ox + bx + bw, fy + oy + by + bh),
                              (128, 255, 128), 1)
                cv2.putText(output, label,
                            (fx + ox + bx, fy + oy + by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 255, 128), 1)

    # --- Ears (Sobel + Contour on face sides) ---
    ear_marker_r = max(6, int(fw * 0.02))
    # Ears are at ~25-55% face height, on the outer edges
    ear_top = int(fh * 0.20)
    ear_bot = int(fh * 0.60)
    ear_width = int(fw * 0.20)

    for side, label in [("left", "L-Ear"), ("right", "R-Ear")]:
        if side == "left":
            ear_roi = face_roi_gray[ear_top:ear_bot, 0:ear_width]
            ear_ox = 0
        else:
            ear_roi = face_roi_gray[ear_top:ear_bot, fw - ear_width:fw]
            ear_ox = fw - ear_width

        ear_found = False
        if ear_roi.shape[0] > 5 and ear_roi.shape[1] > 5:
            # Gaussian Blur + Sobel + Binarization + Contour
            ear_blur = gaussian_blur(ear_roi, ksize=5, tracker=tracker)
            ear_sobel = sobel_edge(ear_blur, tracker=tracker)
            ear_bin = binarization(ear_sobel, thresh=35, tracker=tracker)
            ear_cnts = find_contours(ear_bin, tracker=tracker)

            # Find largest contour in ear region
            if ear_cnts:
                best_ear = max(ear_cnts, key=cv2.contourArea)
                if cv2.contourArea(best_ear) > 50:
                    M_e = cv2.moments(best_ear)
                    if M_e["m00"] > 0:
                        ecx = int(M_e["m10"] / M_e["m00"])
                        ecy = int(M_e["m01"] / M_e["m00"])
                        ec = (fx + ear_ox + ecx, fy + ear_top + ecy)
                        features[label] = ec
                        cv2.circle(output, ec, ear_marker_r, (255, 128, 0), 2)
                        cv2.putText(output, label,
                                    (ec[0] - 20, ec[1] - ear_marker_r - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (255, 128, 0), 1)
                        ear_found = True

        # 耳朵偵測不到就不標記

    print("\n--- Facial Features ---")
    for name, pos in features.items():
        print(f"  {name}: ({pos[0]}, {pos[1]})")

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect_pupils(image_path):
    """Main detection pipeline."""
    tracker = ToolTracker()

    img = cv2.imread(image_path)
    if img is None and HAS_PIL:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            print(f"Loaded via Pillow (unsupported format for OpenCV)")
        except Exception:
            pass
    if img is None:
        print(f"Error: Cannot read image '{image_path}'")
        print("Supported: jpg, png, bmp, webp (install Pillow for avif, etc.)")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()
    img_h, img_w = gray.shape[:2]

    # ===== Stage 1: Multi-angle face detection =====
    result = detect_face_multi_angle(gray)

    left_center_abs = None
    right_center_abs = None
    left_radius = 0
    right_radius = 0
    angle = 0
    rotated_gray = gray
    corrected_gray = gray
    face = None
    eyes_in_face = []

    if result is not None:
        face, angle, rotated_gray = result
        fx, fy, fw, fh = face

        if angle != 0:
            print(f"Face found after rotating {angle} deg")

        # ===== Stage 2: Perspective Transform for tilt correction =====
        corrected_gray, face, eyes_in_face = correct_eye_tilt(
            rotated_gray, face, tracker
        )
        fx, fy, fw, fh = face

        # ===== Stage 3: Reference Pt =====
        landmarks = [
            FakeLandmark(fx + int(fw * 0.18), fy + int(fh * 0.32)),
            FakeLandmark(fx + int(fw * 0.38), fy + int(fh * 0.32)),
            FakeLandmark(fx + int(fw * 0.62), fy + int(fh * 0.32)),
            FakeLandmark(fx + int(fw * 0.82), fy + int(fh * 0.32)),
        ]
        reference_pt(landmarks, [0, 1], tracker=tracker)
        reference_pt(landmarks, [2, 3], tracker=tracker)

        # ===== Stage 4: Sobel on eye region =====
        ery = max(0, fy + int(fh * 0.15))
        ere = min(ery + int(fh * 0.35), corrected_gray.shape[0])
        eye_region = corrected_gray[ery:ere, fx:fx + fw]
        if eye_region.shape[0] > 5 and eye_region.shape[1] > 5:
            sobel_edge(eye_region, tracker=tracker)

        # ===== Stage 5: Pupil detection using eye cascade ROIs =====
        if len(eyes_in_face) < 2:
            # Re-detect eyes
            face_roi = corrected_gray[fy:fy + fh, fx:fx + fw]
            eyes_in_face = detect_eyes_in_face(face_roi)

        if len(eyes_in_face) >= 1:
            # Use eye cascade results as precise ROIs
            for i, (ex, ey, ew, eh) in enumerate(eyes_in_face[:2]):
                abs_ex = fx + ex
                abs_ey = fy + ey
                eye_roi = corrected_gray[abs_ey:abs_ey + eh, abs_ex:abs_ex + ew]

                min_r = max(4, int(min(ew, eh) * 0.08))
                pupil = detect_pupil_in_roi(eye_roi, tracker, min_pupil_r=min_r)

                if pupil is not None:
                    pcx, pcy, pr = pupil
                    abs_pcx = abs_ex + pcx
                    abs_pcy = abs_ey + pcy
                    if i == 0:
                        left_center_abs = (abs_pcx, abs_pcy)
                        left_radius = pr
                    else:
                        right_center_abs = (abs_pcx, abs_pcy)
                        right_radius = pr

            # If only 1 eye found, try proportion-based for the other
            if len(eyes_in_face) == 1 and (left_center_abs is None or right_center_abs is None):
                found_side = "left" if eyes_in_face[0][0] < fw // 2 else "right"
                other_side = "right" if found_side == "left" else "left"
                if other_side == "left":
                    oex = fx + int(fw * 0.10)
                    oew = int(fw * 0.35)
                else:
                    oex = fx + int(fw * 0.55)
                    oew = int(fw * 0.35)
                oey = fy + int(fh * 0.25)
                oeh = int(fh * 0.20)
                oex = max(0, oex)
                oey = max(0, oey)
                oew = min(oew, corrected_gray.shape[1] - oex)
                oeh = min(oeh, corrected_gray.shape[0] - oey)
                oroi = corrected_gray[oey:oey + oeh, oex:oex + oew]
                min_r = max(4, int(min(oew, oeh) * 0.08))
                opupil = detect_pupil_in_roi(oroi, tracker, min_pupil_r=min_r)
                if opupil is not None:
                    pcx, pcy, pr = opupil
                    if other_side == "left":
                        left_center_abs = (oex + pcx, oey + pcy)
                        left_radius = pr
                    else:
                        right_center_abs = (oex + pcx, oey + pcy)
                        right_radius = pr
        else:
            # Fallback: use face proportion ROIs
            for side in ["left", "right"]:
                if side == "left":
                    ex = fx + int(fw * 0.10)
                    ew = int(fw * 0.35)
                else:
                    ex = fx + int(fw * 0.55)
                    ew = int(fw * 0.35)
                ey = fy + int(fh * 0.25)
                eh = int(fh * 0.20)
                ex = max(0, ex)
                ey = max(0, ey)
                ew = min(ew, corrected_gray.shape[1] - ex)
                eh = min(eh, corrected_gray.shape[0] - ey)
                eye_roi = corrected_gray[ey:ey + eh, ex:ex + ew]
                min_r = max(4, int(min(ew, eh) * 0.08))
                pupil = detect_pupil_in_roi(eye_roi, tracker, min_pupil_r=min_r)
                if pupil is not None:
                    pcx, pcy, pr = pupil
                    if side == "left":
                        left_center_abs = (ex + pcx, ey + pcy)
                        left_radius = pr
                    else:
                        right_center_abs = (ex + pcx, ey + pcy)
                        right_radius = pr

        # Transform back to original image if rotated
        if angle != 0:
            center = (img_w // 2, img_h // 2)
            M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)

            def transform_back(px, py):
                pt = np.array([px, py, 1.0])
                new_pt = M_inv @ pt
                return int(new_pt[0]), int(new_pt[1])

            if left_center_abs:
                left_center_abs = transform_back(*left_center_abs)
            if right_center_abs:
                right_center_abs = transform_back(*right_center_abs)

    else:
        # ===== Fallback: direct eye detection =====
        print("No face detected, trying direct eye detection...")
        all_eyes = []
        for cascade_name in ["haarcascade_eye.xml",
                             "haarcascade_lefteye_2splits.xml",
                             "haarcascade_righteye_2splits.xml"]:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + cascade_name
            )
            if cascade.empty():
                continue
            eyes = cascade.detectMultiScale(gray, 1.05, 3, minSize=(15, 15))
            if len(eyes) > 0:
                all_eyes.extend(eyes.tolist())
            if len(all_eyes) >= 2:
                break

        if len(all_eyes) >= 2:
            all_eyes = sorted(all_eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            all_eyes = sorted(all_eyes, key=lambda e: e[0])

            for i, (ex, ey, ew, eh) in enumerate(all_eyes):
                roi = gray[ey:ey + eh, ex:ex + ew]
                pupil = detect_pupil_in_roi(roi, tracker)
                if pupil is not None:
                    pcx, pcy, pr = pupil
                    if i == 0:
                        left_center_abs = (ex + pcx, ey + pcy)
                        left_radius = pr
                    else:
                        right_center_abs = (ex + pcx, ey + pcy)
                        right_radius = pr
        else:
            print("Error: no face or eyes detected")
            tracker.report()
            sys.exit(1)

    # ===== Stage 6: Facial feature detection =====
    if face is not None:
        if angle != 0:
            center = (img_w // 2, img_h // 2)
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_color = cv2.warpAffine(output, M_rot, (img_w, img_h))
            detect_facial_features(
                rotated_gray, face, eyes_in_face, rotated_color, tracker
            )
            M_inv_full = cv2.getRotationMatrix2D(center, -angle, 1.0)
            output = cv2.warpAffine(rotated_color, M_inv_full, (img_w, img_h))
        else:
            detect_facial_features(gray, face, eyes_in_face, output, tracker)

    # ===== Draw pupil results =====
    if left_center_abs is not None:
        cv2.circle(output, left_center_abs, left_radius, (0, 255, 0), 2)
        cv2.circle(output, left_center_abs, 2, (0, 0, 255), -1)
        print(f"L-Pupil: center={left_center_abs}, radius={left_radius}")
    else:
        print("L-Pupil: not detected")

    if right_center_abs is not None:
        cv2.circle(output, right_center_abs, right_radius, (0, 255, 0), 2)
        cv2.circle(output, right_center_abs, 2, (0, 0, 255), -1)
        print(f"R-Pupil: center={right_center_abs}, radius={right_radius}")
    else:
        print("R-Pupil: not detected")

    # ===== Calculate inter-pupil distance =====
    if left_center_abs and right_center_abs:
        dist = math.hypot(
            right_center_abs[0] - left_center_abs[0],
            right_center_abs[1] - left_center_abs[1],
        )
        print(f"Inter-pupil distance: {dist:.2f} px")
        cv2.line(output, left_center_abs, right_center_abs, (255, 0, 0), 1)
        mid = (
            (left_center_abs[0] + right_center_abs[0]) // 2,
            (left_center_abs[1] + right_center_abs[1]) // 2 - 10,
        )
        cv2.putText(output, f"{dist:.1f}px", mid,
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
        print("Cannot calculate distance: at least one pupil not detected")

    # ===== Save result =====
    result_path = "result.jpg"
    cv2.imwrite(result_path, output)
    print(f"Result saved to: {result_path}")

    print()
    tracker.report()
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pupil_detection.py <image_path>")
        sys.exit(1)
    detect_pupils(sys.argv[1])
