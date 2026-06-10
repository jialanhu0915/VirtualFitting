"""
虚拟试衣系统 - 自动关键点检测版本
使用深度学习库自动检测人体和服装关键点
支持Docker环境运行
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe 导入成功")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"❌ MediaPipe 导入失败: {e}")
    print("⚠️  请重新安装: pip install --force-reinstall mediapipe")


class VirtualTryOnSystem:
    """虚拟试衣主系统"""
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe 未正确安装，无法继续")
        
        try:
            # 使用MediaPipe的新API（tasks API）
            # 先尝试旧API
            try:
                from mediapipe.python.solutions import pose as mp_pose
                from mediapipe.python.solutions import drawing_utils as mp_drawing
                
                self.mp_pose = mp_pose
                self.mp_drawing = mp_drawing
                self.pose = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5
                )
                print("✅ MediaPipe Pose 初始化成功（旧API）")
                
            except (ImportError, AttributeError):
                # 如果旧API失败，使用新API
                print("⚠️  尝试使用MediaPipe Tasks API...")
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                # 创建PoseLandmarker选项
                model_path = self._download_pose_model()
                
                options = vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=model_path),
                    output_segmentation_masks=True
                )
                
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
                self.use_new_api = True
                print("✅ MediaPipe Pose 初始化成功（新API）")
                
        except Exception as e:
            raise RuntimeError(f"MediaPipe Pose 初始化失败: {e}")
    
    def _download_pose_model(self):
        """下载MediaPipe Pose模型"""
        import urllib.request
        import os
        
        model_path = "/tmp/pose_landmarker.task"
        if not os.path.exists(model_path):
            print("📥 正在下载Pose模型...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            urllib.request.urlretrieve(url, model_path)
            print("✅ 模型下载完成")
        
        return model_path
        
    def detect_human_keypoints(self, person_image_path):
        """
        自动检测人体关键点（使用MediaPipe Pose）
        
        Args:
            person_image_path: 人体图像路径
            
        Returns:
            keypoints: 人体关键点字典
            person_img: 读取的人体图像
        """
        print("🔍 正在检测人体关键点...")
        
        # 读取图像
        person_img = cv2.imread(person_image_path)
        person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # 检测关键点
        results = self.pose.process(person_rgb)
        
        if not results.pose_landmarks:
            raise ValueError("未检测到人体关键点，请确保图像中有人体")
        
        # 提取关键点坐标
        h, w = person_img.shape[:2]
        keypoints = {}
        
        # MediaPipe关键点索引
        keypoint_names = {
            0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
            4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
            7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
            11: 'left_shoulder', 12: 'right_shoulder',  # 肩膀（重要）
            13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist',  # 手腕
            17: 'left_pinky', 18: 'right_pinky',
            19: 'left_index', 20: 'right_index',
            21: 'left_thumb', 22: 'right_thumb',
            23: 'left_hip', 24: 'right_hip',  # 臀部（重要）
            25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle',
            29: 'left_heel', 30: 'right_heel',
            31: 'left_foot_index', 32: 'right_foot_index'
        }
        
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in keypoint_names:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                visibility = landmark.visibility
                keypoints[keypoint_names[idx]] = {
                    'x': x, 'y': y, 'visibility': visibility
                }
        
        print(f"✅ 检测到 {len(keypoints)} 个人体关键点")
        
        # 可视化关键点
        self._visualize_human_keypoints(person_img, results)
        
        return keypoints, person_img
    
    def detect_clothing_keypoints_auto(self, clothing_image_path):
        """
        自动检测服装关键点（基于图像分析和轮廓检测）
        
        Args:
            clothing_image_path: 服装图像路径
            
        Returns:
            clothing_keypoints: 服装关键点字典
            clothing_img: 读取的服装图像
        """
        print("👕 正在检测服装关键点...")
        
        # 读取服装图像
        clothing_img = cv2.imread(clothing_image_path, cv2.IMREAD_UNCHANGED)
        if clothing_img.shape[2] == 4:  # 如果有alpha通道
            clothing_rgb = cv2.cvtColor(clothing_img, cv2.COLOR_BGRA2RGB)
        else:
            clothing_rgb = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2RGB)
        
        # 1. 服装分割（假设背景干净）
        mask = self._segment_clothing(clothing_rgb)
        
        # 2. 获取服装轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            raise ValueError("未检测到服装区域")
        
        # 获取最大轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 3. 自动提取关键点
        clothing_keypoints = self._extract_clothing_keypoints(
            main_contour, clothing_rgb.shape[:2]
        )
        
        print(f"✅ 检测到 {len(clothing_keypoints)} 个服装关键点")
        
        # 可视化关键点
        self._visualize_clothing_keypoints(clothing_rgb, clothing_keypoints)
        
        return clothing_keypoints, clothing_rgb
    
    def _segment_clothing(self, clothing_img):
        """服装分割 - 基于颜色和边缘"""
        # 转换到HSV空间
        hsv = cv2.cvtColor(clothing_img, cv2.COLOR_RGB2HSV)
        
        # 方法1：基于亮度（假设背景较亮）
        gray = cv2.cvtColor(clothing_img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _extract_clothing_keypoints(self, contour, img_shape):
        """
        从服装轮廓自动提取关键点
        
        策略：
        1. 找到轮廓的上下左右边界
        2. 根据几何特征定义关键点位置
        """
        h, w = img_shape
        
        # 获取轮廓的所有点
        points = contour.reshape(-1, 2)
        
        # 找到边界点
        top_point = points[np.argmin(points[:, 1])]  # 最上面的点
        bottom_point = points[np.argmax(points[:, 1])]  # 最下面的点
        left_point = points[np.argmin(points[:, 0])]  # 最左边的点
        right_point = points[np.argmax(points[:, 0])]  # 最右边的点
        
        # 计算服装的宽度和高度
        clothing_width = right_point[0] - left_point[0]
        clothing_height = bottom_point[1] - top_point[1]
        
        # 自动定义关键点
        keypoints = {
            'top_center': {
                'x': int(top_point[0]),
                'y': int(top_point[1]),
                'description': '领口中心'
            },
            'bottom_center': {
                'x': int(bottom_point[0]),
                'y': int(bottom_point[1]),
                'description': '衣服底部中心'
            },
            'left_shoulder': {
                'x': int(left_point[0] + clothing_width * 0.1),
                'y': int(top_point[1] + clothing_height * 0.15),
                'description': '左肩'
            },
            'right_shoulder': {
                'x': int(right_point[0] - clothing_width * 0.1),
                'y': int(top_point[1] + clothing_height * 0.15),
                'description': '右肩'
            },
            'left_armpit': {
                'x': int(left_point[0] + clothing_width * 0.05),
                'y': int(top_point[1] + clothing_height * 0.3),
                'description': '左腋下'
            },
            'right_armpit': {
                'x': int(right_point[0] - clothing_width * 0.05),
                'y': int(top_point[1] + clothing_height * 0.3),
                'description': '右腋下'
            },
            'left_bottom': {
                'x': int(left_point[0] + clothing_width * 0.15),
                'y': int(bottom_point[1] - clothing_height * 0.05),
                'description': '左下摆'
            },
            'right_bottom': {
                'x': int(right_point[0] - clothing_width * 0.15),
                'y': int(bottom_point[1] - clothing_height * 0.05),
                'description': '右下摆'
            }
        }
        
        return keypoints
    
    def _visualize_human_keypoints(self, img, results):
        """可视化人体关键点"""
        annotated_img = img.copy()
        self.mp_drawing.draw_landmarks(
            annotated_img,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2
            )
        )
        
        # 保存可视化结果
        output_path = "output/human_keypoints_visualization.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"📊 人体关键点可视化已保存: {output_path}")
    
    def _visualize_clothing_keypoints(self, img, keypoints):
        """可视化服装关键点"""
        annotated_img = img.copy()
        
        for name, point in keypoints.items():
            cv2.circle(annotated_img, (point['x'], point['y']), 8, (255, 0, 0), -1)
            cv2.putText(
                annotated_img, name, 
                (point['x'] - 20, point['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )
        
        # 保存可视化结果
        output_path = "output/clothing_keypoints_visualization.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        print(f"📊 服装关键点可视化已保存: {output_path}")
    
    def run(self, person_image_path, clothing_image_path):
        """
        运行完整的虚拟试衣流程
        
        Args:
            person_image_path: 人体图像路径
            clothing_image_path: 服装图像路径
        """
        print("🚀 开始虚拟试衣流程...")
        print("=" * 50)
        
        try:
            # 1. 检测人体关键点
            human_keypoints, person_img = self.detect_human_keypoints(
                person_image_path
            )
            
            # 2. 检测服装关键点
            clothing_keypoints, clothing_img = self.detect_clothing_keypoints_auto(
                clothing_image_path
            )
            
            print("\n" + "=" * 50)
            print("✅ 关键点检测完成！")
            print("\n后续步骤：")
            print("1. TPS变形 - 将服装变形到适合人体的形状")
            print("2. 图像融合 - 将变形的服装融合到人体上")
            
            return human_keypoints, clothing_keypoints, person_img, clothing_img
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            return None


def main():
    """主函数"""
    # 创建虚拟试衣系统
    system = VirtualTryOnSystem()
    
    # 指定图像路径
    person_image = "data_picture/people/image.png"
    clothing_image = "data_picture/clothes/image.png"
    
    # 运行系统
    results = system.run(person_image, clothing_image)


if __name__ == "__main__":
    main()
