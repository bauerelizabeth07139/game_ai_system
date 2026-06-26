"""OCR数值提取引擎 - 基于PaddleOCR PP-OCRv6 Tiny"""
import logging
import re
from typing import Optional, List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# 延迟导入PaddleOCR以支持无OCR环境
_ocr_instance = None


def _get_ocr_instance(use_gpu: bool = False):
    """延迟初始化PaddleOCR实例"""
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=use_gpu,
                show_log=False,
                det_db_thresh=0.3,
                rec_batch_num=1
            )
            logger.info("PaddleOCR初始化成功")
        except Exception as e:
            logger.warning(f"PaddleOCR初始化失败（将禁用OCR功能）: {e}")
            _ocr_instance = "disabled"
    return _ocr_instance


class OCREngine:
    """OCR数值提取引擎"""

    def __init__(self, config: dict = None):
        self.use_gpu = False
        self.enabled = True
        if config:
            self.use_gpu = config.get('use_gpu', False)
            self.enabled = config.get('enabled', True)
        self._ocr = None

    def _ensure_ocr(self):
        """确保OCR实例已初始化"""
        if self._ocr is None:
            if not self.enabled:
                self._ocr = "disabled"
            else:
                self._ocr = _get_ocr_instance(self.use_gpu)
        return self._ocr

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """从图像中提取所有文本"""
        ocr = self._ensure_ocr()
        if ocr == "disabled":
            return []

        try:
            result = ocr.ocr(image, cls=True)
            if not result or not result[0]:
                return []

            texts = []
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                texts.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
            return texts
        except Exception as e:
            logger.error(f"OCR识别异常: {e}")
            return []

    def extract_player_id(self, image: np.ndarray, health_bar_bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """从血条上方区域提取玩家ID/昵称"""
        x, y, w, h = health_bar_bbox
        # 裁剪血条上方区域
        roi_y = max(0, y - h * 2)
        roi_h = y - roi_y
        if roi_h <= 0 or w <= 0:
            return None

        roi = image[roi_y:y, x:x+w]
        if roi.size == 0:
            return None

        texts = self.extract_text(roi)
        if texts:
            # 返回置信度最高的文本
            best = max(texts, key=lambda t: t['confidence'])
            return best['text']
        return None

    def extract_health_number(self, image: np.ndarray, health_bar_bbox: Tuple[int, int, int, int]) -> Optional[int]:
        """从血条右侧区域提取具体血量数字（备用）"""
        x, y, w, h = health_bar_bbox
        # 裁剪血条右侧区域
        img_w = image.shape[1]
        roi_x = min(x + w, img_w)
        roi_w = min(int(w * 0.5), img_w - roi_x)
        if roi_w <= 0 or h <= 0:
            return None

        roi = image[y:y+h, roi_x:roi_x+roi_w]
        if roi.size == 0:
            return None

        texts = self.extract_text(roi)
        for t in texts:
            # 尝试提取数字
            numbers = re.findall(r'\d+', t['text'])
            if numbers:
                try:
                    return int(numbers[0])
                except ValueError:
                    continue
        return None
