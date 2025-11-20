from src.models.CNN.ColorVidNet import ColorVidNet
from src.models.vit.embed import SwinModel
from src.models.CNN.NonlocalNet import WarpNet
from src.models.CNN.FrameColor import frame_colorization
import torch
from src.models.vit.utils import load_params
import os
import cv2
from PIL import Image
from PIL import ImageEnhance as IE
import torchvision.transforms as T
from src.utils import (
    RGB2Lab,
    ToTensor,
    Normalize,
    uncenter_l,
    tensor_lab2rgb
)
import numpy as np
import glob
import argparse
from tqdm import tqdm

def uncenter_ab(ab):
    return ab * 127.0

def rgb_to_grayscale(image):
    """
    將 RGB 圖像轉換為灰階（保持3通道）
    Args:
        image: PIL Image (RGB)
    Returns:
        PIL Image (灰階但保持RGB格式)
    """
    gray = image.convert('L')  # 轉為單通道灰階
    return gray.convert('RGB')  # 轉回3通道RGB格式（但內容是灰階）

class SwinTExCo:
    def __init__(self, weights_path, swin_backbone='swinv2-cr-t-224', device=None):
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.embed_net = SwinModel(pretrained_model=swin_backbone, device=self.device).to(self.device)
        self.nonlocal_net = WarpNet(feature_channel=128).to(self.device)
        self.colornet = ColorVidNet(4).to(self.device)  # ✅ 修改1: 從 7 改為 4
        
        self.embed_net.eval()
        self.nonlocal_net.eval()
        self.colornet.eval()
        
        self.__load_models(self.embed_net, os.path.join(weights_path, "embed_net.pth"))
        self.__load_models(self.nonlocal_net, os.path.join(weights_path, "nonlocal_net.pth"))
        self.__load_models(self.colornet, os.path.join(weights_path, "colornet.pth"))
        
        self.processor = T.Compose([
            T.Resize((224,224)),
            RGB2Lab(),
            ToTensor(),
            Normalize()
        ])
        
    def __load_models(self, model, weight_path):
        params = load_params(weight_path, self.device)
        model.load_state_dict(params, strict=True)
        
    def __preprocess_reference(self, img):
        color_enhancer = IE.Color(img)
        img = color_enhancer.enhance(1.5)
        return img
    
    def __upscale_image(self, large_IA_l, I_current_ab_predict):
        H, W = large_IA_l.shape[2:]
        large_current_ab_predict = torch.nn.functional.interpolate(I_current_ab_predict, 
                                                                size=(H,W), 
                                                                mode="bilinear", 
                                                                align_corners=False)
        large_IA_lab = torch.cat((large_IA_l, uncenter_ab(large_current_ab_predict)), dim=1)
        large_current_rgb_predict = tensor_lab2rgb(large_IA_lab)
        return large_current_rgb_predict.cpu()
    
    def __proccess_sample(self, curr_frame, I_reference_lab, features_B):
        """
        處理單幀圖像 - 單幀上色模式
        Args:
            curr_frame: 當前幀 (PIL Image)
            I_reference_lab: 參考圖的LAB表示
            features_B: 參考圖的特徵
        """
        large_IA_lab = ToTensor()(RGB2Lab()(curr_frame)).unsqueeze(0)
        large_IA_l = large_IA_lab[:, 0:1, :, :].to(self.device)
        
        IA_lab = self.processor(curr_frame)
        IA_lab = IA_lab.unsqueeze(0).to(self.device)
        IA_l = IA_lab[:, 0:1, :, :]
        
        # ✅ 修改2: 移除 placeholder 生成

        with torch.no_grad():
            # ✅ 修改3: frame_colorization 調用移除 placeholder 參數，接收 similarity_map
            I_current_ab_predict, _, similarity_map = frame_colorization(
                IA_l,
                I_reference_lab,
                features_B,
                self.embed_net,
                self.nonlocal_net,
                self.colornet,
                luminance_noise=0,
                temperature=1e-10,
                joint_training=False
            )

        IA_predict_rgb = self.__upscale_image(large_IA_l, I_current_ab_predict)
        IA_predict_rgb = (IA_predict_rgb.squeeze(0).cpu().numpy() * 255.)
        IA_predict_rgb = np.clip(IA_predict_rgb, 0, 255).astype(np.uint8)

        # 返回彩色图像和 similarity_map（用于 FlowChroma Fusion）
        return IA_predict_rgb, similarity_map
    
    def predict_video(self, video, ref_image):
        """
        視頻上色 - 單幀模式 (每幀獨立處理)
        """
        ref_image = self.__preprocess_reference(ref_image)
        
        IB_lab = self.processor(ref_image)
        IB_lab = IB_lab.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            I_reference_lab = IB_lab
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1)).to(self.device)
            features_B = self.embed_net(I_reference_rgb)
        
        while video.isOpened():
            ret, curr_frame = video.read()
            
            if not ret:
                break
            
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            curr_frame = Image.fromarray(curr_frame)
            
            # ✅ 每幀獨立處理,不依賴前一幀
            IA_predict_rgb = self.__proccess_sample(curr_frame, I_reference_lab, features_B)
            
            IA_predict_rgb = IA_predict_rgb.transpose(1,2,0)
            
            yield IA_predict_rgb

        video.release()

    def predict_image(self, image, ref_image):
        """
        單張圖像上色
        """
        ref_image = self.__preprocess_reference(ref_image)

        IB_lab = self.processor(ref_image)
        IB_lab = IB_lab.unsqueeze(0).to(self.device)

        with torch.no_grad():
            I_reference_lab = IB_lab
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1)).to(self.device)
            features_B = self.embed_net(I_reference_rgb)

        curr_frame = image
        IA_predict_rgb, similarity_map = self.__proccess_sample(curr_frame, I_reference_lab, features_B)

        IA_predict_rgb = IA_predict_rgb.transpose(1,2,0)

        return IA_predict_rgb

    def predict_dataset(self, input_root, output_root, image_extensions=['*.jpg', '*.png', '*.jpeg']):
        """
        批量處理數據集
        數據集結構：
        input_root/
            scene1/
                frame_001.jpg
                frame_002.jpg
                ...
            scene2/
                frame_001.jpg
                ...

        處理方式：
        - 每個場景的第一幀作為參考圖像（彩色）
        - 每一幀先轉為灰階，然後使用參考圖上色
        - 保存到 output_root，保持相同的目錄結構和文件名

        Args:
            input_root: 輸入根目錄路徑
            output_root: 輸出根目錄路徑
            image_extensions: 支持的圖像格式列表
        """
        # 獲取所有場景資料夾
        scene_folders = sorted([d for d in os.listdir(input_root)
                               if os.path.isdir(os.path.join(input_root, d))])

        print(f"找到 {len(scene_folders)} 個場景資料夾")

        # 處理每個場景
        for scene_name in tqdm(scene_folders, desc="處理場景"):
            scene_input_path = os.path.join(input_root, scene_name)
            scene_output_path = os.path.join(output_root, scene_name)

            # 創建輸出目錄
            os.makedirs(scene_output_path, exist_ok=True)

            # 獲取該場景的所有圖像文件
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(scene_input_path, ext)))
            image_files = sorted(image_files)

            if len(image_files) == 0:
                print(f"警告：場景 {scene_name} 中沒有找到圖像文件，跳過")
                continue

            print(f"\n處理場景 '{scene_name}': {len(image_files)} 張圖像")

            # 讀取第一幀作為參考圖像（彩色）
            ref_image_path = image_files[0]
            ref_image = Image.open(ref_image_path).convert('RGB')
            print(f"  參考圖像: {os.path.basename(ref_image_path)}")

            # 預處理參考圖像並提取特徵（只做一次）
            ref_image_processed = self.__preprocess_reference(ref_image)
            IB_lab = self.processor(ref_image_processed)
            IB_lab = IB_lab.unsqueeze(0).to(self.device)

            with torch.no_grad():
                I_reference_lab = IB_lab
                I_reference_l = I_reference_lab[:, 0:1, :, :]
                I_reference_ab = I_reference_lab[:, 1:3, :, :]
                I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1)).to(self.device)
                features_B = self.embed_net(I_reference_rgb)

            # 處理該場景的每一幀
            for img_path in tqdm(image_files, desc=f"  {scene_name}", leave=False):
                # 讀取圖像
                image = Image.open(img_path).convert('RGB')

                # 轉為灰階
                gray_image = rgb_to_grayscale(image)

                # 使用參考圖上色
                colorized, _ = self.__proccess_sample(gray_image, I_reference_lab, features_B)
                colorized = colorized.transpose(1, 2, 0)

                # 保存結果（保持原始文件名）
                filename = os.path.basename(img_path)
                output_path = os.path.join(scene_output_path, filename)

                result = Image.fromarray(colorized)
                result.save(output_path)

            print(f"  完成場景 '{scene_name}'")

        print(f"\n✅ 全部完成！結果保存在: {output_root}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SwinTExCo 視頻/圖像上色推理')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'dataset'],
                       help='處理模式: single=單張圖像, dataset=批量處理數據集')
    parser.add_argument('--weights', type=str, required=True,
                       help='模型權重路徑')

    # 單張圖像模式參數
    parser.add_argument('--target_image', type=str,
                       help='目標圖像路徑（mode=single時必需）')
    parser.add_argument('--ref_image', type=str,
                       help='參考圖像路徑（mode=single時必需）')
    parser.add_argument('--output', type=str,
                       help='輸出路徑（mode=single時必需）')

    # 數據集模式參數
    parser.add_argument('--input_root', type=str,
                       help='輸入數據集根目錄（mode=dataset時必需）')
    parser.add_argument('--output_root', type=str,
                       help='輸出數據集根目錄（mode=dataset時必需）')

    args = parser.parse_args()

    # 初始化模型
    print(f"載入模型權重: {args.weights}")
    model = SwinTExCo(args.weights)

    if args.mode == 'single':
        # 單張圖像上色模式
        if not all([args.target_image, args.ref_image, args.output]):
            parser.error("--mode single 需要提供 --target_image, --ref_image, --output")

        print(f"載入目標圖像: {args.target_image}")
        print(f"載入參考圖像: {args.ref_image}")

        target_image = Image.open(args.target_image).convert('RGB')
        ref_image = Image.open(args.ref_image).convert('RGB')

        colorized = model.predict_image(target_image, ref_image)
        result = Image.fromarray(colorized)
        result.save(args.output)

        print(f"✅ 單張圖像上色完成！結果保存在: {args.output}")

    elif args.mode == 'dataset':
        # 批量數據集處理模式
        if not all([args.input_root, args.output_root]):
            parser.error("--mode dataset 需要提供 --input_root, --output_root")

        print(f"輸入數據集: {args.input_root}")
        print(f"輸出目錄: {args.output_root}")
        print("開始批量處理...\n")

        model.predict_dataset(args.input_root, args.output_root)



