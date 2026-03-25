"""
人臉瞳孔偵測 (Pupil Detection)
使用限定的影像處理工具偵測瞳孔位置並計算兩眼瞳孔中心距離。
"""

import sys
import math
import cv2
import numpy as np


class ToolTracker:
    """記錄每次執行使用的工具順序。"""

    def __init__(self):
        self.steps = []

    def log(self, tool_name):
        self.steps.append(tool_name)

    def report(self):
        if self.steps:
            print(f"工具使用順序: {' → '.join(self.steps)}")
        else:
            print("未使用任何工具")


# ---------------------------------------------------------------------------
# 影像處理工具 (每個函式對應一個允許使用的工具)
# ---------------------------------------------------------------------------

def gaussian_blur(img, ksize=7, tracker=None):
    """Gaussian Blur — 高斯模糊降噪"""
    if tracker:
        tracker.log("Gaussian Blur")
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def binarization(img, thresh=50, max_val=255, tracker=None):
    """Binarization — 二值化處理"""
    if tracker:
        tracker.log("Binarization")
    _, binary = cv2.threshold(img, thresh, max_val, cv2.THRESH_BINARY_INV)
    return binary


def sobel_edge(img, tracker=None):
    """Sobel — Sobel 邊緣偵測"""
    if tracker:
        tracker.log("Sobel")
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(np.clip(magnitude, 0, 255))


def canny_edge(img, low=30, high=100, tracker=None):
    """Canny — Canny 邊緣偵測"""
    if tracker:
        tracker.log("Canny")
    return cv2.Canny(img, low, high)


def find_contours(binary_img, tracker=None):
    """Contour — 輪廓偵測"""
    if tracker:
        tracker.log("Contour")
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def hough_circles(img, dp=1.2, min_dist=30, param1=100, param2=20,
                  min_radius=3, max_radius=80, tracker=None):
    """Hough — 霍夫圓偵測"""
    if tracker:
        tracker.log("Hough")
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    return circles


def perspective_transform(img, src_pts, dst_pts, out_size, tracker=None):
    """Perspective Transform — 透視變換"""
    if tracker:
        tracker.log("Perspective Transform")
    M = cv2.getPerspectiveTransform(
        np.float32(src_pts), np.float32(dst_pts)
    )
    return cv2.warpPerspective(img, M, out_size)


def reference_pt(landmarks, indices, tracker=None):
    """Reference Pt — 從人臉特徵點取得參考點座標"""
    if tracker:
        tracker.log("Reference Pt")
    return [(landmarks[i].x, landmarks[i].y) for i in indices]


# ---------------------------------------------------------------------------
# 眼部 ROI 擷取
# ---------------------------------------------------------------------------

def get_eye_roi(gray, face_rect, side="left"):
    """根據人臉框粗略切出眼睛區域 (不依賴 landmark)。"""
    x, y, w, h = face_rect
    eye_h = int(h * 0.28)
    eye_y = y + int(h * 0.18)
    if side == "left":
        eye_x = x + int(w * 0.08)
        eye_w = int(w * 0.42)
    else:
        eye_x = x + int(w * 0.50)
        eye_w = int(w * 0.42)
    # 防止越界
    eye_x = max(0, eye_x)
    eye_y = max(0, eye_y)
    eye_w = min(eye_w, gray.shape[1] - eye_x)
    eye_h = min(eye_h, gray.shape[0] - eye_y)
    roi = gray[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w]
    return roi, (eye_x, eye_y, eye_w, eye_h)


# ---------------------------------------------------------------------------
# 瞳孔偵測核心管線
# ---------------------------------------------------------------------------

def detect_pupil_in_roi(roi, tracker, eye_label="eye"):
    """
    在眼部 ROI 中偵測瞳孔。
    回傳 (center_x_in_roi, center_y_in_roi, radius) 或 None。
    """
    h, w = roi.shape[:2]
    if h < 5 or w < 5:
        return None

    # 1) Gaussian Blur — 降噪 + 去除反光
    blurred = gaussian_blur(roi, ksize=7, tracker=tracker)

    # 2) Binarization — 瞳孔為暗色區域，二值化分離
    #    動態計算閾值：取 ROI 最暗 20% 的平均值作為基準
    sorted_px = np.sort(blurred.flatten())
    dark_mean = int(sorted_px[:max(1, len(sorted_px) // 5)].mean())
    thresh = min(dark_mean + 30, 80)
    binary = binarization(blurred, thresh=thresh, tracker=tracker)

    # 3) Canny — 邊緣偵測強化瞳孔輪廓
    edges = canny_edge(blurred, low=30, high=80, tracker=tracker)

    # 4) 合併 binary 與 edges
    combined = cv2.bitwise_or(binary, edges)

    # 5) Contour — 尋找候選輪廓
    contours = find_contours(combined, tracker=tracker)

    # 6) 從 contours 篩選最佳瞳孔候選
    best = None
    best_score = -1
    center_region_x = w / 2
    center_region_y = h / 2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
        if radius < 2 or radius > min(w, h) / 2:
            continue
        # 圓度 (越接近 1 越圓)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        # 位置分數 (越靠近 ROI 中心越好)
        dist_from_center = math.hypot(cx - center_region_x, cy - center_region_y)
        max_dist = math.hypot(center_region_x, center_region_y)
        position_score = 1 - (dist_from_center / max_dist) if max_dist > 0 else 0
        # 綜合分數
        score = circularity * 0.5 + position_score * 0.5
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), int(radius))

    # 7) Hough 圓偵測作為備援 / 驗證
    min_r = max(3, int(min(w, h) * 0.05))
    max_r = max(min_r + 1, int(min(w, h) * 0.45))
    circles = hough_circles(
        blurred, dp=1.5, min_dist=w // 3,
        param1=80, param2=18,
        min_radius=min_r, max_radius=max_r,
        tracker=tracker,
    )

    hough_best = None
    if circles is not None:
        best_h_score = -1
        for c in circles[0]:
            cx, cy, r = c
            dist = math.hypot(cx - center_region_x, cy - center_region_y)
            pos_s = 1 - (dist / max_dist) if max_dist > 0 else 0
            if pos_s > best_h_score:
                best_h_score = pos_s
                hough_best = (int(cx), int(cy), int(r))

    # 決策: 優先 contour，Hough 備援
    if best is not None:
        return best
    if hough_best is not None:
        return hough_best
    return None


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def detect_pupils(image_path):
    """偵測圖片中的瞳孔並輸出結果。"""
    tracker = ToolTracker()

    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤: 無法讀取圖片 '{image_path}'")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()

    # --- 人臉偵測 (使用 Haar Cascade) ---
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # 多角度嘗試偵測
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # 嘗試放寬參數
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    if len(faces) == 0:
        print("錯誤: 未偵測到人臉")
        tracker.report()
        sys.exit(1)

    # 選最大的臉
    face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = face

    # --- 對眼部 ROI 做透視校正 (處理斜視/低頭) ---
    eye_region_y = fy + int(fh * 0.15)
    eye_region_h = int(fh * 0.35)
    eye_region = gray[eye_region_y:eye_region_y + eye_region_h, fx:fx + fw]

    if eye_region.shape[0] > 10 and eye_region.shape[1] > 10:
        er_h, er_w = eye_region.shape[:2]
        src_pts = [[0, 0], [er_w, 0], [er_w, er_h], [0, er_h]]
        dst_pts = [[0, 0], [er_w, 0], [er_w, er_h], [0, er_h]]
        eye_region_corrected = perspective_transform(
            eye_region, src_pts, dst_pts, (er_w, er_h), tracker=tracker
        )
    else:
        eye_region_corrected = eye_region

    # --- 使用 Reference Pt 標記眼角作為參考 ---
    # 利用人臉幾何比例推算眼角參考點
    class FakeLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = [
        FakeLandmark(fx + int(fw * 0.18), fy + int(fh * 0.32)),  # 左眼左角
        FakeLandmark(fx + int(fw * 0.38), fy + int(fh * 0.32)),  # 左眼右角
        FakeLandmark(fx + int(fw * 0.62), fy + int(fh * 0.32)),  # 右眼左角
        FakeLandmark(fx + int(fw * 0.82), fy + int(fh * 0.32)),  # 右眼右角
    ]
    ref_left = reference_pt(landmarks, [0, 1], tracker=tracker)
    ref_right = reference_pt(landmarks, [2, 3], tracker=tracker)

    # --- 偵測左眼瞳孔 ---
    left_roi, (lx, ly, lw, lh) = get_eye_roi(gray, face, side="left")
    left_pupil = detect_pupil_in_roi(left_roi, tracker, eye_label="left")

    # --- 偵測右眼瞳孔 ---
    right_roi, (rx, ry, rw, rh) = get_eye_roi(gray, face, side="right")
    right_pupil = detect_pupil_in_roi(right_roi, tracker, eye_label="right")

    # --- Sobel 輔助驗證 (對整個眼區做 Sobel 確認邊緣一致性) ---
    if eye_region_corrected.shape[0] > 5 and eye_region_corrected.shape[1] > 5:
        sobel_edge(eye_region_corrected, tracker=tracker)

    # --- 繪製結果 ---
    left_center_abs = None
    right_center_abs = None

    if left_pupil is not None:
        cx, cy, r = left_pupil
        abs_cx, abs_cy = lx + cx, ly + cy
        left_center_abs = (abs_cx, abs_cy)
        cv2.circle(output, (abs_cx, abs_cy), r, (0, 255, 0), 2)
        cv2.circle(output, (abs_cx, abs_cy), 2, (0, 0, 255), -1)
        print(f"左眼瞳孔: 中心=({abs_cx}, {abs_cy}), 半徑={r}")
    else:
        print("左眼瞳孔: 未偵測到")

    if right_pupil is not None:
        cx, cy, r = right_pupil
        abs_cx, abs_cy = rx + cx, ry + cy
        right_center_abs = (abs_cx, abs_cy)
        cv2.circle(output, (abs_cx, abs_cy), r, (0, 255, 0), 2)
        cv2.circle(output, (abs_cx, abs_cy), 2, (0, 0, 255), -1)
        print(f"右眼瞳孔: 中心=({abs_cx}, {abs_cy}), 半徑={r}")
    else:
        print("右眼瞳孔: 未偵測到")

    # --- 計算距離 ---
    if left_center_abs and right_center_abs:
        dist = math.hypot(
            right_center_abs[0] - left_center_abs[0],
            right_center_abs[1] - left_center_abs[1],
        )
        print(f"兩眼瞳孔中心距離: {dist:.2f} 像素")
        # 在圖上畫連線
        cv2.line(output, left_center_abs, right_center_abs, (255, 0, 0), 1)
        mid = (
            (left_center_abs[0] + right_center_abs[0]) // 2,
            (left_center_abs[1] + right_center_abs[1]) // 2 - 10,
        )
        cv2.putText(output, f"{dist:.1f}px", mid,
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
        print("無法計算距離: 至少一眼未偵測到瞳孔")

    # --- 儲存結果 ---
    result_path = "result.jpg"
    cv2.imwrite(result_path, output)
    print(f"結果已儲存至: {result_path}")

    # --- 輸出工具使用順序 ---
    print()
    tracker.report()

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方式: python pupil_detection.py <image_path>")
        sys.exit(1)
    detect_pupils(sys.argv[1])
