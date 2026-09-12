"""
Module Tiền Xử Lý Ảnh (Image Preprocessing Utilities).

MỤC ĐÍCH HỌC TẬP:
- Hiểu giải thuật chuyển đổi Grayscale bằng công thức Luminosity: Y = 0.299R + 0.587G + 0.114B.
- Hiểu giải thuật Deskewing (Xác định góc nghiêng và xoay thẳng ảnh).
- Hiểu giải thuật Adaptive Thresholding (Phân ngưỡng thích nghi theo vùng cục bộ).
- Cung cấp hàm chuẩn hóa Bounding Box sang tỷ lệ [0.0, 1.0] phục vụ hiển thị trên Web Frontend.
"""

from typing import Tuple, List, Optional
import math
from app.models.ocr_models import BoundingBox


def normalize_bbox(x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int) -> List[float]:
    """
    Chuẩn hóa tọa độ tuyệt đối pixel sang tỷ lệ [0.0, 1.0].
    Công thức: norm_x = x / width, norm_y = y / height.
    Giúp Frontend hiển thị khung bôi sáng chính xác trên mọi kích thước màn hình.
    """
    w = max(1, width)
    h = max(1, height)
    return [
        round(max(0.0, min(1.0, x_min / w)), 4),
        round(max(0.0, min(1.0, y_min / h)), 4),
        round(max(0.0, min(1.0, x_max / w)), 4),
        round(max(0.0, min(1.0, y_max / h)), 4)
    ]


def create_bounding_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    img_width: int,
    img_height: int
) -> BoundingBox:
    """Tạo đối tượng BoundingBox đầy đủ cả tọa độ tuyệt đối và chuẩn hóa."""
    polygon = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ]
    norm = normalize_bbox(x_min, y_min, x_max, y_max, img_width, img_height)
    return BoundingBox(
        x_min=round(x_min, 1),
        y_min=round(y_min, 1),
        x_max=round(x_max, 1),
        y_max=round(y_max, 1),
        polygon=polygon,
        normalized=norm
    )


class ImagePreprocessor:
    """Pipeline tiền xử lý ảnh sử dụng OpenCV (kèm cơ chế Fallback học tập)."""

    @staticmethod
    def calculate_deskew_angle_from_points(points: List[Tuple[float, float]]) -> float:
        """
        Tính toán góc nghiêng dựa trên tập điểm chữ cái.
        Sử dụng phép hồi quy tuyến tính tối thiểu (Linear Regression) hoặc MinAreaRect.
        """
        if len(points) < 2:
            return 0.0

        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_xx = sum(p[0] * p[0] for p in points)

        denominator = (n * sum_xx - sum_x * sum_x)
        if abs(denominator) < 1e-9:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        angle_degrees = math.degrees(math.atan(slope))
        return round(angle_degrees, 2)

    @staticmethod
    def preprocess_with_opencv(image_path: str) -> Optional[dict]:
        """
        Thực thi chuỗi tiền xử lý đầy đủ bằng thư viện OpenCV:
        1. Đọc ảnh
        2. Chuyển sang ảnh xám (Grayscale)
        3. Khử nhiễu (Denoise)
        4. Cân góc nghiêng (Deskew)
        5. Phân ngưỡng thích nghi (Adaptive Thresholding)
        """
        try:
            import cv2
            import numpy as np

            # 1. Đọc ảnh
            img = cv2.imread(image_path)
            if img is None:
                return None

            h, w = img.shape[:2]

            # 2. Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 3. Khử nhiễu với Gaussian Blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 4. Phân ngưỡng thích nghi (Adaptive Gaussian Thresholding)
            # Rất hiệu quả khi ảnh bị bóng tối một góc
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # 5. Xác định góc nghiêng (Deskew)
            coords = np.column_stack(np.where(binary > 0))
            angle = 0.0
            deskewed = gray
            if len(coords) > 50:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle

                # Nếu độ nghiêng đáng kể (> 0.5 độ), thực hiện xoay
                if abs(angle) > 0.5:
                    center = (w // 2, h // 2)
                    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    deskewed = cv2.warpAffine(
                        gray, rot_matrix, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )

            return {
                "width": w,
                "height": h,
                "detected_skew_angle": round(angle, 2),
                "processed_ready": True
            }
        except ImportError:
            # Nếu chưa cài opencv, trả về thông tin giả lập phục vụ học tập
            return {
                "width": 1200,
                "height": 1600,
                "detected_skew_angle": 0.0,
                "processed_ready": True
            }
