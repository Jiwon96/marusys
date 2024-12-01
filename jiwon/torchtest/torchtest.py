import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import sys
from utils import Padding  # 필요한 클래스/함수 import

def load_audio(audio_path, sample_rate=16000):
    """
    오디오 파일을 로드하고 전처리하는 함수
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform

def load_model_and_predict(model_path, audio_path, device='cpu' if torch.cuda.is_available() else 'cpu'):
    """
    저장된 PyTorch 모델을 로드하고 오디오 예측을 수행하는 함수
    
    Args:
        model_path (str): .pt 파일 경로
        audio_path (str): 오디오 파일 경로
        device (str): 사용할 디바이스 (cuda 또는 cpu)
    
    Returns:
        예측 결과
    """
    # 모델 로드
    model = torch.jit.load(model_path)  # TorchScript 모델 로드
    model.to(device)
    model.eval()
    
    # 오디오 데이터 로드 및 전처리
    waveform = load_audio(audio_path)
    
    padding = Padding()
    waveform = padding(waveform)

    # 전처리된 데이터를 모델 입력 형식에 맞게 변환
    input_batch = waveform.to(device)
    
    # 예측 수행
    with torch.no_grad():
        output = model(input_batch)
    
    return output

# 사용 예시
if __name__ == "__main__":
    premodel_path = "/home/ssafy/bcresnet-main/torchtest/logmel_optimized.ptl"
    model_path = "/home/ssafy/bcresnet-main/torchtest/model_optimized.ptl"
    audio_path = "/home/ssafy/bcresnet-main/data/speech_commands_v0.02/train_12class/hey_ssafy"
    model = torch.jit.load(model_path)  # TorchScript 모델 로드
    model.to('cpu')
    model.eval()
    # 예측 수행
    predictions = load_model_and_predict(premodel_path, audio_path)
    print(predictions.shape)
    predictions = predictions.unsqueeze(1) 
    print(predictions.shape)
    with torch.no_grad():
        output = model(predictions)

    # 결과 처리
    probabilities = F.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {probabilities[predicted_class]:.4f}")

    # 클래스 레이블이 있는 경우 (예시)
    classes = label_dict = [
    "_silence_",
    "_unknown_",
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes",
    "hey_ssafy",
    "others"
    ]  # 실제 클래스 목록으로 수정 필요
    if predicted_class < len(classes):
        print(f"Predicted label: {classes[predicted_class]}")
