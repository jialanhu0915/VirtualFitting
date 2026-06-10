"""
虚拟试衣系统 - 简化稳定版本
提供多种关键点检测方案，确保系统可运行
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


class SimpleVirtualTryOn:
    """简化的虚拟试衣系统"""
    
    def __init__(self):
        print("🚀 初始化虚拟试衣系统...")
        
        # 创建输出目录
        os.makedirs("output", exist_ok=True)
        
        # 检测可用的关键点检测方法
        self.detection_method = self._detect_available_method()
        
    def _detect_available_method(self):
        """检测可用的关键点检测方法"""
        methods = []
        
        # 尝试MediaPipe
        try:
            import mediapipe as mp
            print("✅ MediaPipe可用")
            methods.append('mediapipe')
        except:
            print("⚠️  MediaPipe不可用")
        
        # OpenCV始终可用
        methods.append('opencv')
        print("✅ OpenCV可用")
        
        return methods[0] if methods else 'manual'
    
    def detect_human_keypoints_simple(self, image_path):
        """
        简化的人体关键点检测
        使用OpenCV的Haar级联分类器检测人脸和人体
        """
        print(f"🔍 检测人体关键点: {image_path}")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        h, w = img.shape[:2]
        
        # 尝试使用Haar级联检测人脸
        keypoints = {}
        face_detected = False
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 尝试多个可能的级联文件路径
            cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
                cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            ]
            
            faces = []
            for cascade_path in cascade_paths:
                if os.path.exists(cascade_path):
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        print(f"✅ 使用级联文件检测到人脸: {os.path.basename(cascade_path)}")
                        face_detected = True
                        break
        except Exception as e:
            print(f"⚠️  人脸检测失败: {e}")
            face_detected = False
        
        if face_detected and len(faces) > 0:
            # 使用检测到的人脸位置推算其他关键点
            (fx, fy, fw, fh) = faces[0]
            
            # 脸部中心
            keypoints['face'] = {
                'x': int(fx + fw/2),
                'y': int(fy + fh/2),
                'description': '脸部中心'
            }
            
            # 根据脸部位置估算其他关键点
            # 这些是基于人体比例的估算
            head_size = fh
            body_top = fy + fh
            
            # 肩膀位置（估算）
            keypoints['left_shoulder'] = {
                'x': int(fx - fw * 0.3),
                'y': int(body_top + head_size * 0.5),
                'description': '左肩'
            }
            
            keypoints['right_shoulder'] = {
                'x': int(fx + fw * 1.3),
                'y': int(body_top + head_size * 0.5),
                'description': '右肩'
            }
            
            # 臀部位置（估算）
            keypoints['left_hip'] = {
                'x': int(fx + fw * 0.2),
                'y': int(body_top + head_size * 3.0),
                'description': '左臀'
            }
            
            keypoints['right_hip'] = {
                'x': int(fx + fw * 0.8),
                'y': int(body_top + head_size * 3.0),
                'description': '右臀'
            }
            
            print("✅ 检测到人体关键点（基于人脸检测）")
        else:
            # 如果没检测到人脸，使用图像中心估算
            print("⚠️  未检测到人脸，使用图像中心估算")
            center_x, center_y = w // 2, h // 2
            
            keypoints = {
                'face': {'x': center_x, 'y': center_y - h//4, 'description': '脸部中心'},
                'left_shoulder': {'x': center_x - w//6, 'y': center_y - h//8, 'description': '左肩'},
                'right_shoulder': {'x': center_x + w//6, 'y': center_y - h//8, 'description': '右肩'},
                'left_hip': {'x': center_x - w//8, 'y': center_y + h//4, 'description': '左臀'},
                'right_hip': {'x': center_x + w//8, 'y': center_y + h//4, 'description': '右臀'}
            }
        
        # 可视化
        self._visualize_keypoints(img, keypoints, "output/human_keypoints.jpg")
        
        return keypoints, img
    
    def detect_clothing_keypoints_auto(self, clothing_path):
        """
        自动检测服装关键点
        使用图像处理技术
        """
        print(f"👕 检测服装关键点: {clothing_path}")
        
        # 读取服装图像
        img = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图像: {clothing_path}")
        
        if len(img.shape) == 3 and img.shape[2] == 4:
            # 有alpha通道，转换为RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为灰度图
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # 二值化分割
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("未检测到服装区域")
        
        # 获取最大轮廓
        main_contour = max(contours, key=cv2.contourArea)
        points = main_contour.reshape(-1, 2)
        
        h, w = rgb_img.shape[:2]
        
        # 提取关键点
        top_point = points[np.argmin(points[:, 1])]
        bottom_point = points[np.argmax(points[:, 1])]
        left_point = points[np.argmin(points[:, 0])]
        right_point = points[np.argmax(points[:, 0])]
        
        clothing_width = right_point[0] - left_point[0]
        clothing_height = bottom_point[1] - top_point[1]
        
        keypoints = {
            'top_center': {
                'x': int(top_point[0]),
                'y': int(top_point[1]),
                'description': '领口中心'
            },
            'bottom_center': {
                'x': int(bottom_point[0]),
                'y': int(bottom_point[1]),
                'description': '衣服底部'
            },
            'left_shoulder': {
                'x': int(left_point[0] + clothing_width * 0.15),
                'y': int(top_point[1] + clothing_height * 0.2),
                'description': '左肩'
            },
            'right_shoulder': {
                'x': int(right_point[0] - clothing_width * 0.15),
                'y': int(top_point[1] + clothing_height * 0.2),
                'description': '右肩'
            },
            'left_armpit': {
                'x': int(left_point[0] + clothing_width * 0.1),
                'y': int(top_point[1] + clothing_height * 0.35),
                'description': '左腋下'
            },
            'right_armpit': {
                'x': int(right_point[0] - clothing_width * 0.1),
                'y': int(top_point[1] + clothing_height * 0.35),
                'description': '右腋下'
            }
        }
        
        print("✅ 检测到服装关键点")
        
        # 可视化
        self._visualize_keypoints(rgb_img, keypoints, "output/clothing_keypoints.jpg")
        
        return keypoints, rgb_img
    
    def _visualize_keypoints(self, img, keypoints, output_path):
        """可视化关键点"""
        annotated = img.copy()
        
        for name, point in keypoints.items():
            x, y = point['x'], point['y']
            
            # 画圆
            cv2.circle(annotated, (x, y), 10, (0, 255, 0), -1)
            
            # 标注文字
            cv2.putText(
                annotated, 
                f"{name}", 
                (x - 30, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
        
        # 保存
        if len(annotated.shape) == 3:
            cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(output_path, annotated)
        
        print(f"📊 关键点可视化已保存: {output_path}")
    
    def run(self, person_path, clothing_path):
        """运行完整的检测流程"""
        print("\n" + "="*60)
        print("🚀 虚拟试衣系统启动")
        print("="*60)
        
        try:
            # 1. 检测人体关键点
            print("\n[步骤1/2] 检测人体关键点")
            human_kpts, person_img = self.detect_human_keypoints_simple(person_path)
            
            # 2. 检测服装关键点
            print("\n[步骤2/2] 检测服装关键点")
            clothing_kpts, clothing_img = self.detect_clothing_keypoints_auto(clothing_path)
            
            # 3. 输出结果
            print("\n" + "="*60)
            print("✅ 关键点检测完成！")
            print("="*60)
            
            print("\n📋 检测到的人体关键点:")
            for name, point in human_kpts.items():
                print(f"  • {point['description']}: ({point['x']}, {point['y']})")
            
            print("\n📋 检测到的服装关键点:")
            for name, point in clothing_kpts.items():
                print(f"  • {point['description']}: ({point['x']}, {point['y']})")
            
            print("\n📁 输出文件:")
            print("  • output/human_keypoints.jpg - 人体关键点可视化")
            print("  • output/clothing_keypoints.jpg - 服装关键点可视化")
            
            print("\n🎯 下一步：")
            print("  1. 实现TPS变形算法")
            print("  2. 实现图像融合算法")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    # 图像路径
    person_image = "data_picture/people/image.png"
    clothing_image = "data_picture/clothes/image.png"
    
    # 检查文件是否存在
    if not os.path.exists(person_image):
        print(f"❌ 文件不存在: {person_image}")
        return
    
    if not os.path.exists(clothing_image):
        print(f"❌ 文件不存在: {clothing_image}")
        return
    
    # 创建系统并运行
    system = SimpleVirtualTryOn()
    system.run(person_image, clothing_image)


if __name__ == "__main__":
    main()
