import os

import onnxruntime as rt
import numpy as np
import torch
import torch.nn.functional as F
import cv2

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.onnx')
VOCAB_PATH = os.path.join(os.path.dirname(__file__), 'vocab.txt')

def _load_label_mapping(filepath):
    label_mapping = dict()
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        cnt = 2
        for line in lines:
            line = line.strip('\n')
            label_mapping[cnt] = line
            cnt += 1
    return label_mapping


def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found or could not be read.")
    return image


def preprocess(image):
    # 预处理图像：调整大小、归一化等
    target_height = 32
    target_width = 640
    h, w, _ = image.shape
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h))

    # Create a new image with the target size and fill with black (or any other color)
    padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    padded_img[:new_h, :new_w, :] = resized_img

    padded_img = padded_img.astype(np.float32) / 255.0
    padded_img = np.transpose(padded_img, (2, 0, 1))  # HWC to CHW
    padded_img = np.expand_dims(padded_img, axis=0)  # 添加批次维度
    return padded_img


class ONNXModel:
    def __init__(self):
        # 加载ONNX模型
        self.sess = rt.InferenceSession(MODEL_PATH)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        # 加载标签映射（如果提供的话）
        self.label_mapping = _load_label_mapping(VOCAB_PATH) if os.path.exists(VOCAB_PATH) else None

    def postprocess(self, preds):
        outprobs = F.softmax(torch.tensor(preds), dim=-1)
        preds = torch.argmax(outprobs, -1).cpu().numpy()

        if self.label_mapping:
            batch_size, length = preds.shape
            final_str_list = []
            for i in range(batch_size):
                pred_idx = preds[i].tolist()
                last_p = 0
                str_pred = []
                for p in pred_idx:
                    if p != last_p and p != 0:
                        str_pred.append(self.label_mapping[p])
                    last_p = p
                final_str = ''.join(str_pred)
                final_str_list.append(final_str)
            return final_str_list
        return preds

    def predict(self, image_path):
        image = read_image(image_path)
        input_data = preprocess(image)
        preds = self.sess.run([self.output_name], {self.input_name: input_data})
        results = self.postprocess(preds[0])
        return results
