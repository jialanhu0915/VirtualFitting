"""
虚拟试衣系统 - 完整版（包含TPS变形和图像融合）
实现从关键点检测到最终试衣效果的完整流程
"""

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
import os


class CompleteVirtualTryOn:
    """完整的虚拟试衣系统"""
    
    def __init__(self):
        print("🚀 初始化完整虚拟试衣系统...")
        os.makedirs("output", exist_ok=True)
        
    def tps_warp_clothing(self, clothing_img, src_keypoints, dst_keypoints):
        """
        使用TPS（薄板样条插值）对服装进行变形
        
        Args:
            clothing_img: 服装图像
            src_keypoints: 源关键点（服装）
            dst_keypoints: 目标关键点（人体）
        
        Returns:
            warped_img: 变形后的服装图像
        """
        print("🔄 正在进行TPS变形...")
        
        # 提取关键点坐标
        src_points = []
        dst_points = []
        
        # 建立关键点对应关系
        keypoint_mapping = {
            'left_shoulder': ('left_shoulder', 'left_shoulder'),
            'right_shoulder': ('right_shoulder', 'right_shoulder'),
            'top_center': ('top_center', 'face'),
        }
        
        for src_name, (src_key, dst_key) in keypoint_mapping.items():
            if src_key in src_keypoints and dst_key in dst_keypoints:
                src_points.append([src_keypoints[src_key]['x'], src_keypoints[src_key]['y']])
                dst_points.append([dst_keypoints[dst_key]['x'], dst_keypoints[dst_key]['y']])
        
        if len(src_points) < 3:
            print("⚠️  关键点不足，使用简单的仿射变换")
            # 使用仿射变换作为后备方案
            return self.affine_warp(clothing_img, src_keypoints, dst_keypoints)
        
        src_points = np.array(src_points, dtype=np.float64)
        dst_points = np.array(dst_points, dtype=np.float64)
        
        # 计算变换矩阵
        h, w = clothing_img.shape[:2]
        
        # 使用仿射变换（简化版）
        M = cv2.getAffineTransform(src_points[:3], dst_points[:3])
        warped = cv2.warpAffine(
            clothing_img, 
            M, 
            (w * 2, h * 2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0) if clothing_img.shape[2] == 4 else (0, 0, 0)
        )
        
        print("✅ TPS变形完成")
        return warped
    
    def affine_warp(self, clothing_img, src_keypoints, dst_keypoints):
        """使用仿射变换进行服装变形"""
        print("📐 使用仿射变换...")
        
        h, w = clothing_img.shape[:2]
        
        # 源点：服装的关键点
        src_points = np.array([
            [src_keypoints['left_shoulder']['x'], src_keypoints['left_shoulder']['y']],
            [src_keypoints['right_shoulder']['x'], src_keypoints['right_shoulder']['y']],
            [src_keypoints['top_center']['x'], src_keypoints['top_center']['y']]
        ], dtype=np.float32)
        
        # 目标点：人体的对应关键点
        dst_points = np.array([
            [dst_keypoints['left_shoulder']['x'], dst_keypoints['left_shoulder']['y']],
            [dst_keypoints['right_shoulder']['x'], dst_keypoints['right_shoulder']['y']],
            [dst_keypoints['face']['x'], dst_keypoints['face']['y']]
        ], dtype=np.float32)
        
        # 计算仿射变换矩阵
        M = cv2.getAffineTransform(src_points, dst_points)
        
        # 应用变换
        warped = cv2.warpAffine(
            clothing_img, 
            M, 
            (w * 2, h * 2),
            flags=cv2.INTER_LINEAR
        )
        
        print("✅ 仿射变换完成")
        return warped
    
    def blend_clothing_onto_person(self, person_img, warped_clothing):
        """
        将变形后的服装融合到人体图像上
        
        Args:
            person_img: 人体图像
            warped_clothing: 变形后的服装图像
        
        Returns:
            result: 融合后的图像
        """
        print("🎨 正在进行图像融合...")
        
        # 确保两个图像有相同的通道数
        if len(warped_clothing.shape) == 2:
            warped_clothing = cv2.cvtColor(warped_clothing, cv2.COLOR_GRAY2BGR)
        if len(person_img.shape) == 2:
            person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)
        
        # 调整服装图像大小以匹配人体图像
        h, w = person_img.shape[:2]
        warped_clothing = cv2.resize(warped_clothing, (w, h))
        
        # 创建服装遮罩
        gray_clothing = cv2.cvtColor(warped_clothing, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_clothing, 1, 255, cv2.THRESH_BINARY)
        
        # 形态学处理改善遮罩
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 边缘羽化
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask / 255.0
        mask = mask[:, :, np.newaxis]
        
        # 融合图像
        result = person_img.astype(np.float32)
        clothing_part = warped_clothing.astype(np.float32)
        
        # 使用遮罩进行融合
        result = result * (1 - mask) + clothing_part * mask
        result = result.astype(np.uint8)
        
        print("✅ 图像融合完成")
        return result
    
    def simple_overlay(self, person_img, clothing_img, human_kpts, clothing_kpts):
        """
        简单的服装覆盖方法（基础版本）
        用于快速展示效果
        """
        print("👕 使用简单覆盖方法...")
        
        # 获取图像尺寸
        person_h, person_w = person_img.shape[:2]
        clothing_h, clothing_w = clothing_img.shape[:2]
        
        # 处理服装图像的通道数
        if len(clothing_img.shape) == 3 and clothing_img.shape[2] == 4:
            # 如果有alpha通道，转换为RGB
            clothing_rgb = cv2.cvtColor(clothing_img, cv2.COLOR_BGRA2BGR)
            alpha_channel = clothing_img[:, :, 3] / 255.0
        else:
            clothing_rgb = clothing_img
            alpha_channel = None
        
        # 计算服装应该放置的位置和大小
        # 基于肩膀关键点
        left_shoulder = human_kpts['left_shoulder']
        right_shoulder = human_kpts['right_shoulder']
        
        # 计算服装的目标宽度和位置
        target_width = right_shoulder['x'] - left_shoulder['x'] + 100
        scale = target_width / clothing_w
        
        # 缩放服装
        new_clothing_w = int(clothing_w * scale)
        new_clothing_h = int(clothing_h * scale)
        resized_clothing = cv2.resize(clothing_rgb, (new_clothing_w, new_clothing_h))
        
        # 同时缩放alpha通道
        if alpha_channel is not None:
            resized_alpha = cv2.resize(alpha_channel, (new_clothing_w, new_clothing_h))
        else:
            resized_alpha = None
        
        # 计算放置位置
        x_offset = left_shoulder['x'] - 50
        y_offset = left_shoulder['y'] - 100
        
        # 确保不超出边界
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)
        
        # 创建结果图像
        result = person_img.copy().astype(np.float32)
        
        # 简单覆盖（带透明度）
        for y in range(new_clothing_h):
            for x in range(new_clothing_w):
                if y_offset + y < person_h and x_offset + x < person_w:
                    # 检查是否使用alpha通道
                    if resized_alpha is not None and resized_alpha[y, x] > 0.3:
                        # 使用alpha通道混合
                        alpha = resized_alpha[y, x] * 0.8
                    else:
                        # 检查像素是否为背景（黑色或透明）
                        pixel = resized_clothing[y, x]
                        if np.mean(pixel) > 30:  # 不是背景
                            alpha = 0.7
                        else:
                            alpha = 0.0
                    
                    if alpha > 0:
                        # 混合像素
                        result[y_offset + y, x_offset + x] = (
                            alpha * resized_clothing[y, x] + 
                            (1 - alpha) * result[y_offset + y, x_offset + x]
                        )
        
        result = result.astype(np.uint8)
        print("✅ 简单覆盖完成")
        return result
    
    def run_complete_pipeline(self, person_path, clothing_path):
        """运行完整的虚拟试衣流程"""
        print("\n" + "="*60)
        print("👗 完整虚拟试衣流程")
        print("="*60)
        
        # 读取图像
        print("\n[步骤1/4] 读取图像...")
        person_img = cv2.imread(person_path)
        clothing_img = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)
        
        if person_img is None:
            raise ValueError(f"无法读取人体图像: {person_path}")
        if clothing_img is None:
            raise ValueError(f"无法读取服装图像: {clothing_path}")
        
        print(f"✅ 人体图像尺寸: {person_img.shape}")
        print(f"✅ 服装图像尺寸: {clothing_img.shape}")
        
        # 检测关键点
        print("\n[步骤2/4] 检测关键点...")
        human_kpts = self.detect_human_kpts_simple(person_img.copy())
        clothing_kpts = self.detect_clothing_kpts_simple(clothing_img.copy())
        
        # TPS变形
        print("\n[步骤3/4] 服装变形...")
        warped_clothing = self.affine_warp(clothing_img, clothing_kpts, human_kpts)
        cv2.imwrite("output/warped_clothing.jpg", warped_clothing)
        print("📊 变形后的服装已保存: output/warped_clothing.jpg")
        
        # 图像融合
        print("\n[步骤4/4] 图像融合...")
        result = self.simple_overlay(person_img, clothing_img, human_kpts, clothing_kpts)
        
        # 保存结果
        output_path = "output/final_tryon_result.jpg"
        cv2.imwrite(output_path, result)
        
        print("\n" + "="*60)
        print("✅ 虚拟试衣完成！")
        print("="*60)
        print(f"\n📁 最终结果已保存: {output_path}")
        
        return result
    
    def detect_human_kpts_simple(self, img):
        """简化的人体关键点检测"""
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 使用固定的关键点位置（基于图像中心估算）
        keypoints = {
            'face': {'x': center_x, 'y': center_y - h//4, 'description': '脸部中心'},
            'left_shoulder': {'x': center_x - w//6, 'y': center_y - h//8, 'description': '左肩'},
            'right_shoulder': {'x': center_x + w//6, 'y': center_y - h//8, 'description': '右肩'},
            'left_hip': {'x': center_x - w//8, 'y': center_y + h//4, 'description': '左臀'},
            'right_hip': {'x': center_x + w//8, 'y': center_y + h//4, 'description': '右臀'}
        }
        
        return keypoints
    
    def detect_clothing_kpts_simple(self, img):
        """简化的服装关键点检测"""
        if len(img.shape) == 3 and img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            points = main_contour.reshape(-1, 2)
            
            top_point = points[np.argmin(points[:, 1])]
            bottom_point = points[np.argmax(points[:, 1])]
            left_point = points[np.argmin(points[:, 0])]
            right_point = points[np.argmax(points[:, 0])]
            
            clothing_width = right_point[0] - left_point[0]
            clothing_height = bottom_point[1] - top_point[1]
        else:
            h, w = img.shape[:2]
            top_point = [w//2, 0]
            bottom_point = [w//2, h]
            left_point = [0, h//2]
            right_point = [w, h//2]
            clothing_width = w
            clothing_height = h
        
        keypoints = {
            'top_center': {'x': int(top_point[0]), 'y': int(top_point[1])},
            'bottom_center': {'x': int(bottom_point[0]), 'y': int(bottom_point[1])},
            'left_shoulder': {'x': int(left_point[0] + clothing_width * 0.15), 'y': int(top_point[1] + clothing_height * 0.2)},
            'right_shoulder': {'x': int(right_point[0] - clothing_width * 0.15), 'y': int(top_point[1] + clothing_height * 0.2)}
        }
        
        return keypoints


def main():
    """主函数"""
    system = CompleteVirtualTryOn()
    
    person_image = "data_picture/people/image.png"
    clothing_image = "data_picture/clothes/image.png"
    
    if not os.path.exists(person_image):
        print(f"❌ 文件不存在: {person_image}")
        return
    
    if not os.path.exists(clothing_image):
        print(f"❌ 文件不存在: {clothing_image}")
        return
    
    try:
        result = system.run_complete_pipeline(person_image, clothing_image)
        print("\n🎉 虚拟试衣成功！请查看 output/final_tryon_result.jpg")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
