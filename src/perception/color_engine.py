"""颜色逻辑视觉引擎 - 基于OpenCV HSV颜色检测"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ColorEngine:
    """基于HSV颜色空间的视觉检测引擎"""

    # 固定阈值常量
    RED_LOWER_1 = np.array([0, 50, 50])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([170, 50, 50])
    RED_UPPER_2 = np.array([180, 255, 255])
    GREEN_LOWER = np.array([35, 50, 50])
    GREEN_UPPER = np.array([85, 255, 255])

    # 几何筛选参数
    MIN_AREA = 50
    MAX_AREA = 2000
    MIN_ASPECT_RATIO = 2.0
    MAX_ASPECT_RATIO = 6.0
    Y_POSITION_RATIO = 0.6  # 仅取屏幕上半部分

    def __init__(self, config: dict = None):
        if config:
            hsv = config.get('hsv', {})
            self.RED_LOWER_1 = np.array(hsv.get('red_lower_1', [0, 50, 50]))
            self.RED_UPPER_1 = np.array(hsv.get('red_upper_1', [10, 255, 255]))
            self.RED_LOWER_2 = np.array(hsv.get('red_lower_2', [170, 50, 50]))
            self.RED_UPPER_2 = np.array(hsv.get('red_upper_2', [180, 255, 255]))
            self.GREEN_LOWER = np.array(hsv.get('green_lower', [35, 50, 50]))
            self.GREEN_UPPER = np.array(hsv.get('green_upper', [85, 255, 255]))
            geo = config.get('geometry', {})
            self.MIN_AREA = geo.get('min_area', 50)
            self.MAX_AREA = geo.get('max_area', 2000)
            self.MIN_ASPECT_RATIO = geo.get('min_aspect_ratio', 2.0)
            self.MAX_ASPECT_RATIO = geo.get('max_aspect_ratio', 6.0)
            self.Y_POSITION_RATIO = geo.get('y_position_ratio', 0.6)

    def resize_image(self, image: np.ndarray, short_side: int = 480) -> np.ndarray:
        """等比例缩放图片至短边为指定像素"""
        h, w = image.shape[:2]
        if h < w:
            scale = short_side / h
        else:
            scale = short_side / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _create_red_mask(self, hsv: np.ndarray) -> np.ndarray:
        """创建红色掩膜（双阈值合并）"""
        mask1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        mask2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        return cv2.bitwise_or(mask1, mask2)

    def _create_green_mask(self, hsv: np.ndarray) -> np.ndarray:
        """创建绿色掩膜（队友）"""
        return cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)

    def _filter_contours(self, contours: list, h: int) -> List[dict]:
        """几何筛选轮廓：面积、长宽比、位置"""
        results = []
        y_limit = int(h * self.Y_POSITION_RATIO)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if y >= y_limit:  # 仅取屏幕上半部分
                continue

            if bh == 0:
                continue
            aspect_ratio = bw / bh
            if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
                continue

            results.append({
                'bbox': (x, y, bw, bh),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'center': (x + bw // 2, y + bh // 2)
            })

        return results

    def _calculate_health(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """计算血量：红色像素宽度/(红色+灰色背景)宽度"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        if roi.size == 0 or w == 0:
            return 0.0

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = self._create_red_mask(hsv_roi)

        # 灰色背景检测（低饱和度区域）
        gray_lower = np.array([0, 0, 50])
        gray_upper = np.array([180, 50, 200])
        gray_mask = cv2.inRange(hsv_roi, gray_lower, gray_upper)

        # 按列统计
        red_cols = np.any(red_mask > 0, axis=0)
        gray_cols = np.any(gray_mask > 0, axis=0)
        combined_cols = red_cols | gray_cols

        red_width = np.sum(red_cols)
        total_width = np.sum(combined_cols)

        if total_width == 0:
            return 0.0

        return float(np.clip(red_width / total_width, 0.0, 1.0))

    def process(self, image: np.ndarray) -> dict:
        """处理单帧图像，返回检测结果

        Returns:
            dict: {
                'enemies': [{'bbox': ..., 'center': ..., 'health': float, 'area': ...}],
                'allies': [{'bbox': ..., 'center': ..., 'health': float, 'area': ...}],
                'image_size': (w, h)
            }
        """
        try:
            resized = self.resize_image(image)
            h, w = resized.shape[:2]

            # 转HSV
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

            # 检测敌人（红色）
            red_mask = self._create_red_mask(hsv)
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            enemies = self._filter_contours(red_contours, h)
            for e in enemies:
                e['health'] = self._calculate_health(resized, e['bbox'])
                e['type'] = 'enemy'

            # 检测队友（绿色）
            green_mask = self._create_green_mask(hsv)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            allies = self._filter_contours(green_contours, h)
            for a in allies:
                a['health'] = self._calculate_health(resized, a['bbox'])
                a['type'] = 'ally'

            return {
                'enemies': enemies,
                'allies': allies,
                'image_size': (w, h)
            }
        except Exception as e:
            logger.error(f"颜色引擎处理异常: {e}")
            return {'enemies': [], 'allies': [], 'image_size': (0, 0)}
