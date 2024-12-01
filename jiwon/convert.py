import torch
import torch.nn as nn
import torchaudio
import warnings
from torch.utils.mobile_optimizer import optimize_for_mobile
warnings.filterwarnings("ignore", category=UserWarning)

class LogMel(nn.Module):
    def __init__(
        self, sample_rate=16000, hop_length=160, 
        win_length=480, n_fft=512, n_mels=40
    ):
        super(LogMel, self).__init__()
        
        torchaudio.set_audio_backend("sox_io")
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, x):
        mel = self.mel_spec(x)
        return (mel + 1e-6).log()

def save_model(model):
    try:
        # PTL 파일로 저장
        print("모델이 성공적으로 저장되었습니다.")
        model.to('cpu')
        # 테스트
        input_sample = torch.randn(16000)
         # TorchScript로 변환
        traced_model = torch.jit.trace(model, input_sample)
        traced_model.to('cpu')  # CPU로 변환된 모델 저장

        traced_model.save("model_scripted.pt")  # 일반 TorchScript 모델 저장
        
        # 모바일 최적화 적용
        optimized_model = optimize_for_mobile(traced_model)
        optimized_model._save_for_lite_interpreter("model_optimized.ptl")  # 모바일 최적화 모델 저장
        print("Optimized model saved as model_optimized.ptl")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    save_model()