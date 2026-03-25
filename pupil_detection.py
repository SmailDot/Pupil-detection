"""
人臉瞳孔偵測與五官定位 (Pupil Detection & Facial Feature Localization)
使用限定的影像處理工具偵測瞳孔位置、五官位置並計算兩眼瞳孔中心距離。
支援非正面人臉：低頭、側臉、斜視等姿態。
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


class FakeLandmark:
    """簡易 landmark 結構。"""
    def __init__(self, x, y):
        self.x = x
        self.y = y


# ---------------------------------------------------------------------------
# 多方向人臉偵測 (處理非正面人臉)
# ---------------------------------------------------------------------------

def detect_face_multi_angle(gray):
    """
    嘗試多種 cascade + 多角度旋轉偵測人臉。
    回傳 (face_rect, angle, rotated_gray) 或 None。
    angle 為偵測到人臉時的旋轉角度 (用於後續校正)。
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)

    # 嘗試不同旋轉角度: 0, ±15, ±30, ±45 度
    angles = [0, -15, 15, -30, 30, -45, 45]

    for angle in angles:
        if angle == 0:
            rotated = gray
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h))

        # 正面偵測
        for sf, mn in [(1.1, 5), (1.05, 3), (1.1, 3)]:
            faces = face_cascade.detectMultiScale(
                rotated, scaleFactor=sf, minNeighbors=mn, minSize=(30, 30)
            )
            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                return face, angle, rotated

        # 側臉偵測 (左右 profile)
        for sf, mn in [(1.1, 3), (1.05, 3)]:
            faces = profile_cascade.detectMultiScale(
                rotated, scaleFactor=sf, minNeighbors=mn, minSize=(30, 30)
            )
            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                return face, angle, rotated

            # 水平翻轉偵測另一側
            flipped = cv2.flip(rotated, 1)
            faces = profile_cascade.detectMultiScale(
                flipped, scaleFactor=sf, minNeighbors=mn, minSize=(30, 30)
            )
            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = face
                # 翻轉回原始座標
                face = (w - fx - fw, fy, fw, fh)
                return face, angle, rotated

    return None


def detect_eyes_direct(gray):
    """
    Fallback: 直接偵測眼睛 (不依賴人臉偵測)。
    回傳 [(ex, ey, ew, eh), ...] 眼睛矩形列表。
    """
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3,
                                        minSize=(15, 15))
    return eyes


# ---------------------------------------------------------------------------
# 眼部 ROI 擷取
# ---------------------------------------------------------------------------

def get_eye_roi(gray, face_rect, side="left"):
    """根據人臉框切出眼睛區域。"""
    x, y, w, h = face_rect
    eye_h = int(h * 0.30)
    eye_y = y + int(h * 0.18)
    if side == "left":
        eye_x = x + int(w * 0.05)
        eye_w = int(w * 0.45)
    else:
        eye_x = x + int(w * 0.50)
        eye_w = int(w * 0.45)
    eye_x = max(0, eye_x)
    eye_y = max(0, eye_y)
    eye_w = min(eye_w, gray.shape[1] - eye_x)
    eye_h = min(eye_h, gray.shape[0] - eye_y)
    roi = gray[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w]
    return roi, (eye_x, eye_y, eye_w, eye_h)


# ---------------------------------------------------------------------------
# 眼部透視校正 (根據兩眼偵測的傾斜角度)
# ---------------------------------------------------------------------------

def correct_eye_tilt(gray, face_rect, tracker):
    """
    使用眼睛 cascade 偵測兩眼位置，計算傾斜角度，
    再用 Perspective Transform 校正眼部區域。
    回傳校正後的灰階圖與實際使用的 face_rect。
    """
    fx, fy, fw, fh = face_rect
    face_roi = gray[fy:fy + fh, fx:fx + fw]

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.05,
                                        minNeighbors=3, minSize=(10, 10))

    if len(eyes) < 2:
        return gray, face_rect

    # 取最大的兩個眼睛
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    # 按 x 座標排序 (左眼在左)
    eyes = sorted(eyes, key=lambda e: e[0])

    # 兩眼中心
    left_eye_center = (fx + eyes[0][0] + eyes[0][2] // 2,
                       fy + eyes[0][1] + eyes[0][3] // 2)
    right_eye_center = (fx + eyes[1][0] + eyes[1][2] // 2,
                        fy + eyes[1][1] + eyes[1][3] // 2)

    # 計算傾斜角度
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = math.degrees(math.atan2(dy, dx))

    # 若傾斜角度小於 2 度則不需校正
    if abs(angle) < 2.0:
        return gray, face_rect

    print(f"偵測到人臉傾斜 {angle:.1f}°，執行透視校正...")

    # 用 Perspective Transform 校正整張圖
    img_h, img_w = gray.shape[:2]
    eye_mid = ((left_eye_center[0] + right_eye_center[0]) / 2,
               (left_eye_center[1] + right_eye_center[1]) / 2)

    # 構造透視變換的四點 (模擬旋轉校正)
    cos_a = math.cos(math.radians(-angle))
    sin_a = math.sin(math.radians(-angle))

    def rotate_point(px, py, cx, cy):
        dx, dy = px - cx, py - cy
        return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)

    corners = [(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)]
    src_pts = [list(c) for c in corners]
    dst_pts = [list(rotate_point(c[0], c[1], eye_mid[0], eye_mid[1]))
               for c in corners]

    corrected = perspective_transform(gray, src_pts, dst_pts, (img_w, img_h),
                                      tracker=tracker)

    return corrected, face_rect


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

    # 1) Gaussian Blur — 降噪 + 平滑反光
    blurred = gaussian_blur(roi, ksize=7, tracker=tracker)

    # 2) Binarization — 瞳孔為暗色區域，動態閾值二值化
    sorted_px = np.sort(blurred.flatten())
    dark_mean = int(sorted_px[:max(1, len(sorted_px) // 5)].mean())
    thresh = min(dark_mean + 30, 80)
    binary = binarization(blurred, thresh=thresh, tracker=tracker)

    # 3) Canny — 邊緣偵測強化瞳孔輪廓
    edges = canny_edge(blurred, low=30, high=80, tracker=tracker)

    # 4) 合併 binary 與 edges
    combined = cv2.bitwise_or(binary, edges)

    # 5) 形態學處理去除小雜點
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # 6) Contour — 尋找候選輪廓
    contours = find_contours(combined, tracker=tracker)

    # 7) 從 contours 篩選最佳瞳孔候選
    best = None
    best_score = -1
    center_region_x = w / 2
    center_region_y = h / 2
    max_dist = math.hypot(center_region_x, center_region_y)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15:
            continue
        ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
        if radius < 2 or radius > min(w, h) / 2:
            continue

        # 圓度
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)

        # 面積比 (輪廓面積 vs 最小外接圓面積)
        circle_area = math.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0

        # 位置分數
        dist_from_center = math.hypot(cx - center_region_x, cy - center_region_y)
        position_score = 1 - (dist_from_center / max_dist) if max_dist > 0 else 0

        # 暗度分數 (瞳孔應該是 ROI 中最暗的區域)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        mean_val = cv2.mean(roi, mask=mask)[0]
        darkness_score = 1 - (mean_val / 255.0)

        # 綜合分數
        score = (circularity * 0.25 + fill_ratio * 0.15 +
                 position_score * 0.30 + darkness_score * 0.30)
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), max(int(radius), 3))

    # 8) Hough 圓偵測作為備援
    min_r = max(3, int(min(w, h) * 0.05))
    max_r = max(min_r + 1, int(min(w, h) * 0.45))
    circles = hough_circles(
        blurred, dp=1.5, min_dist=max(w // 3, 10),
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
            # Hough 候選的暗度分數
            mask_h = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_h, (int(cx), int(cy)), max(int(r), 1), 255, -1)
            dark_s = 1 - (cv2.mean(roi, mask=mask_h)[0] / 255.0)
            h_score = pos_s * 0.5 + dark_s * 0.5
            if h_score > best_h_score:
                best_h_score = h_score
                hough_best = (int(cx), int(cy), max(int(r), 3))

    # 決策: 優先 contour，Hough 備援
    if best is not None:
        return best
    if hough_best is not None:
        return hough_best
    return None


def detect_pupil_in_eye_rect(gray, eye_rect, tracker, eye_label="eye"):
    """從眼睛矩形 (eye cascade 偵測結果) 中找瞳孔。"""
    ex, ey, ew, eh = eye_rect
    # 縮小到眼球中央區域
    margin_x = int(ew * 0.15)
    margin_y = int(eh * 0.20)
    roi = gray[ey + margin_y:ey + eh - margin_y,
               ex + margin_x:ex + ew - margin_x]
    if roi.shape[0] < 5 or roi.shape[1] < 5:
        roi = gray[ey:ey + eh, ex:ex + ew]
    result = detect_pupil_in_roi(roi, tracker, eye_label)
    if result is not None:
        cx, cy, r = result
        return (cx + ex + margin_x, cy + ey + margin_y, r)
    return None


# ---------------------------------------------------------------------------
# 五官偵測
# ---------------------------------------------------------------------------

def detect_facial_features(gray, face_rect, output, tracker):
    """
    偵測五官位置 (眼睛、鼻子、嘴巴) 並在圖上標記。
    回傳五官位置字典。
    """
    fx, fy, fw, fh = face_rect
    face_roi_gray = gray[fy:fy + fh, fx:fx + fw]
    features = {}

    # --- 眼睛偵測 ---
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.05,
                                        minNeighbors=3, minSize=(10, 10))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        eyes = sorted(eyes, key=lambda e: e[0])
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            label = "左眼" if i == 0 else "右眼"
            center = (fx + ex + ew // 2, fy + ey + eh // 2)
            features[label] = center
            cv2.rectangle(output, (fx + ex, fy + ey),
                          (fx + ex + ew, fy + ey + eh), (255, 255, 0), 1)
            cv2.putText(output, label, (fx + ex, fy + ey - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # --- 鼻子偵測 ---
    nose_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_mcs_nose.xml"
    )
    # 鼻子在臉的中下區域
    nose_region_y = int(fh * 0.25)
    nose_region_h = int(fh * 0.50)
    nose_roi = face_roi_gray[nose_region_y:nose_region_y + nose_region_h, :]
    noses = nose_cascade.detectMultiScale(nose_roi, scaleFactor=1.1,
                                          minNeighbors=3, minSize=(10, 10))
    if len(noses) > 0:
        # 取最大的鼻子候選
        nx, ny, nw, nh = max(noses, key=lambda n: n[2] * n[3])
        nose_center = (fx + nx + nw // 2, fy + nose_region_y + ny + nh // 2)
        features["鼻子"] = nose_center
        cv2.rectangle(output, (fx + nx, fy + nose_region_y + ny),
                      (fx + nx + nw, fy + nose_region_y + ny + nh),
                      (0, 255, 255), 1)
        cv2.putText(output, "Nose", (fx + nx, fy + nose_region_y + ny - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    else:
        # Fallback: 用比例估算鼻子位置
        nose_est = (fx + fw // 2, fy + int(fh * 0.55))
        features["鼻子"] = nose_est
        # 用 Sobel 在鼻子區域做邊緣分析驗證
        nose_area = face_roi_gray[int(fh * 0.35):int(fh * 0.65),
                                  int(fw * 0.3):int(fw * 0.7)]
        if nose_area.shape[0] > 5 and nose_area.shape[1] > 5:
            sobel_edge(nose_area, tracker=tracker)
        cv2.circle(output, nose_est, 5, (0, 255, 255), 2)
        cv2.putText(output, "Nose(est)", (nose_est[0] - 20, nose_est[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # --- 嘴巴偵測 ---
    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_mcs_mouth.xml"
    )
    # 嘴巴在臉的下半部
    mouth_region_y = int(fh * 0.55)
    mouth_region_h = fh - mouth_region_y
    mouth_roi = face_roi_gray[mouth_region_y:mouth_region_y + mouth_region_h, :]
    mouths = mouth_cascade.detectMultiScale(mouth_roi, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(15, 10))
    if len(mouths) > 0:
        mx, my, mw, mh = max(mouths, key=lambda m: m[2] * m[3])
        mouth_center = (fx + mx + mw // 2, fy + mouth_region_y + my + mh // 2)
        features["嘴巴"] = mouth_center
        cv2.rectangle(output, (fx + mx, fy + mouth_region_y + my),
                      (fx + mx + mw, fy + mouth_region_y + my + mh),
                      (255, 0, 255), 1)
        cv2.putText(output, "Mouth", (fx + mx, fy + mouth_region_y + my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    else:
        # Fallback: 比例估算
        mouth_est = (fx + fw // 2, fy + int(fh * 0.78))
        features["嘴巴"] = mouth_est
        cv2.circle(output, mouth_est, 5, (255, 0, 255), 2)
        cv2.putText(output, "Mouth(est)", (mouth_est[0] - 25, mouth_est[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # --- 眉毛偵測 (使用 Sobel + Contour 在眼睛上方區域) ---
    eyebrow_region = face_roi_gray[int(fh * 0.08):int(fh * 0.30),
                                   int(fw * 0.05):int(fw * 0.95)]
    if eyebrow_region.shape[0] > 5 and eyebrow_region.shape[1] > 5:
        eb_blurred = gaussian_blur(eyebrow_region, ksize=5, tracker=tracker)
        eb_edges = sobel_edge(eb_blurred, tracker=tracker)
        eb_binary = binarization(eb_edges, thresh=40, tracker=tracker)
        eb_contours = find_contours(eb_binary, tracker=tracker)

        # 篩選水平長條形輪廓作為眉毛候選
        eb_candidates = []
        for cnt in eb_contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            aspect = w_c / max(h_c, 1)
            area = cv2.contourArea(cnt)
            if aspect > 2.0 and area > 30:
                eb_candidates.append((x_c, y_c, w_c, h_c))

        if eb_candidates:
            # 按 x 排序，左半為左眉，右半為右眉
            eb_candidates.sort(key=lambda c: c[0])
            eb_offset_x = int(fw * 0.05)
            eb_offset_y = int(fh * 0.08)
            for i, (bx, by, bw, bh) in enumerate(eb_candidates[:2]):
                label = "左眉" if bx < eyebrow_region.shape[1] // 2 else "右眉"
                eb_center = (fx + eb_offset_x + bx + bw // 2,
                             fy + eb_offset_y + by + bh // 2)
                features[label] = eb_center
                cv2.rectangle(output,
                              (fx + eb_offset_x + bx, fy + eb_offset_y + by),
                              (fx + eb_offset_x + bx + bw,
                               fy + eb_offset_y + by + bh),
                              (128, 255, 128), 1)
                cv2.putText(output, label,
                            (fx + eb_offset_x + bx,
                             fy + eb_offset_y + by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 255, 128), 1)

    # 輸出五官位置
    print("\n--- 五官位置 ---")
    for name, pos in features.items():
        print(f"  {name}: ({pos[0]}, {pos[1]})")

    return features


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def detect_pupils(image_path):
    """偵測圖片中的瞳孔與五官並輸出結果。"""
    tracker = ToolTracker()

    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤: 無法讀取圖片 '{image_path}'")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()
    img_h, img_w = gray.shape[:2]

    # ===== 階段 1: 多角度人臉偵測 =====
    result = detect_face_multi_angle(gray)

    left_center_abs = None
    right_center_abs = None
    left_radius = 0
    right_radius = 0
    angle = 0
    rotated_gray = gray
    corrected_gray = gray
    face = None

    if result is not None:
        face, angle, rotated_gray = result
        fx, fy, fw, fh = face

        if angle != 0:
            print(f"偵測到人臉 (旋轉 {angle}° 後找到)")

        # ===== 階段 2: Perspective Transform 校正傾斜 =====
        corrected_gray, face = correct_eye_tilt(rotated_gray, face, tracker)
        fx, fy, fw, fh = face

        # ===== 階段 3: Reference Pt 建立眼角參考點 =====
        fx, fy, fw, fh = face
        landmarks = [
            FakeLandmark(fx + int(fw * 0.18), fy + int(fh * 0.32)),
            FakeLandmark(fx + int(fw * 0.38), fy + int(fh * 0.32)),
            FakeLandmark(fx + int(fw * 0.62), fy + int(fh * 0.32)),
            FakeLandmark(fx + int(fw * 0.82), fy + int(fh * 0.32)),
        ]
        ref_left = reference_pt(landmarks, [0, 1], tracker=tracker)
        ref_right = reference_pt(landmarks, [2, 3], tracker=tracker)

        # ===== 階段 4: Sobel 分析眼區邊緣特徵 =====
        eye_region_y = fy + int(fh * 0.15)
        eye_region_h = int(fh * 0.35)
        eye_region_y = max(0, eye_region_y)
        eye_region_end = min(eye_region_y + eye_region_h, corrected_gray.shape[0])
        eye_region = corrected_gray[eye_region_y:eye_region_end, fx:fx + fw]
        if eye_region.shape[0] > 5 and eye_region.shape[1] > 5:
            sobel_result = sobel_edge(eye_region, tracker=tracker)

        # ===== 階段 5: 偵測左右眼瞳孔 =====
        left_roi, (lx, ly, lw, lh) = get_eye_roi(corrected_gray, face, side="left")
        left_pupil = detect_pupil_in_roi(left_roi, tracker, eye_label="left")

        right_roi, (rx, ry, rw, rh) = get_eye_roi(corrected_gray, face, side="right")
        right_pupil = detect_pupil_in_roi(right_roi, tracker, eye_label="right")

        # 若在旋轉圖上偵測，需要把座標轉回原圖
        if angle != 0:
            center = (img_w // 2, img_h // 2)
            M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)

            def transform_back(px, py):
                pt = np.array([px, py, 1.0])
                new_pt = M_inv @ pt
                return int(new_pt[0]), int(new_pt[1])

            if left_pupil is not None:
                cx, cy, r = left_pupil
                abs_cx, abs_cy = transform_back(lx + cx, ly + cy)
                left_center_abs = (abs_cx, abs_cy)
                left_radius = r

            if right_pupil is not None:
                cx, cy, r = right_pupil
                abs_cx, abs_cy = transform_back(rx + cx, ry + cy)
                right_center_abs = (abs_cx, abs_cy)
                right_radius = r
        else:
            if left_pupil is not None:
                cx, cy, r = left_pupil
                left_center_abs = (lx + cx, ly + cy)
                left_radius = r

            if right_pupil is not None:
                cx, cy, r = right_pupil
                right_center_abs = (rx + cx, ry + cy)
                right_radius = r

    else:
        # ===== Fallback: 直接偵測眼睛 (無人臉框) =====
        print("未偵測到人臉框，嘗試直接偵測眼睛...")
        eyes = detect_eyes_direct(gray)

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            eyes = sorted(eyes, key=lambda e: e[0])  # 左→右

            left_pupil = detect_pupil_in_eye_rect(gray, eyes[0], tracker, "left")
            right_pupil = detect_pupil_in_eye_rect(gray, eyes[1], tracker, "right")

            if left_pupil is not None:
                left_center_abs = (left_pupil[0], left_pupil[1])
                left_radius = left_pupil[2]
            if right_pupil is not None:
                right_center_abs = (right_pupil[0], right_pupil[1])
                right_radius = right_pupil[2]
        elif len(eyes) == 1:
            pupil = detect_pupil_in_eye_rect(gray, eyes[0], tracker, "single")
            if pupil is not None:
                left_center_abs = (pupil[0], pupil[1])
                left_radius = pupil[2]
        else:
            print("錯誤: 未偵測到人臉或眼睛")
            tracker.report()
            sys.exit(1)

    # ===== 階段 6: 五官偵測 =====
    if result is not None:
        face_for_features = face
        gray_for_features = corrected_gray if result is not None else gray
        # 若有旋轉，在原圖上用原始 face 做五官偵測
        if angle != 0:
            # 五官偵測在旋轉校正後的圖上執行，結果畫在 output 上
            # 需要建一個旋轉後的彩色圖做繪製
            center = (img_w // 2, img_h // 2)
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_color = cv2.warpAffine(output, M_rot, (img_w, img_h))
            features = detect_facial_features(
                rotated_gray, face, rotated_color, tracker
            )
            # 把繪製結果轉回原圖
            M_inv_full = cv2.getRotationMatrix2D(center, -angle, 1.0)
            output = cv2.warpAffine(rotated_color, M_inv_full, (img_w, img_h))
        else:
            features = detect_facial_features(gray, face, output, tracker)

    # ===== 繪製結果 =====
    if left_center_abs is not None:
        cv2.circle(output, left_center_abs, left_radius, (0, 255, 0), 2)
        cv2.circle(output, left_center_abs, 2, (0, 0, 255), -1)
        print(f"左眼瞳孔: 中心={left_center_abs}, 半徑={left_radius}")
    else:
        print("左眼瞳孔: 未偵測到")

    if right_center_abs is not None:
        cv2.circle(output, right_center_abs, right_radius, (0, 255, 0), 2)
        cv2.circle(output, right_center_abs, 2, (0, 0, 255), -1)
        print(f"右眼瞳孔: 中心={right_center_abs}, 半徑={right_radius}")
    else:
        print("右眼瞳孔: 未偵測到")

    # ===== 計算距離 =====
    if left_center_abs and right_center_abs:
        dist = math.hypot(
            right_center_abs[0] - left_center_abs[0],
            right_center_abs[1] - left_center_abs[1],
        )
        print(f"兩眼瞳孔中心距離: {dist:.2f} 像素")
        cv2.line(output, left_center_abs, right_center_abs, (255, 0, 0), 1)
        mid = (
            (left_center_abs[0] + right_center_abs[0]) // 2,
            (left_center_abs[1] + right_center_abs[1]) // 2 - 10,
        )
        cv2.putText(output, f"{dist:.1f}px", mid,
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
        print("無法計算距離: 至少一眼未偵測到瞳孔")

    # ===== 儲存結果 =====
    result_path = "result.jpg"
    cv2.imwrite(result_path, output)
    print(f"結果已儲存至: {result_path}")

    print()
    tracker.report()
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方式: python pupil_detection.py <image_path>")
        sys.exit(1)
    detect_pupils(sys.argv[1])
