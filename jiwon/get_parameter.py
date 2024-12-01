import torch
from pathlib import Path
import numpy as np
import sys
from datetime import datetime

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def analyze_model(model_path, log_path=None):
    """
    저장된 PyTorch 모델의 상세 정보를 분석하고 결과를 파일로 저장합니다.
    
    Args:
        model_path (str): .pt 파일 경로
        log_path (str): 로그 파일 저장 경로 (기본값: None)
    """
    # 로그 파일 경로 설정
    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"model_analysis_{timestamp}.txt"
    
    # 로그 리디렉션 설정
    sys.stdout = Logger(log_path)
    
    try:
        # 모델 로드
        if model_path.endswith('.pt'):
            model = torch.load(model_path, map_location='cpu')
        elif model_path.endswith('.pth'):
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            return analyze_state_dict(state_dict)
        else:
            print(f"지원하지 않는 파일 형식입니다: {model_path}")
            return None
            
        print(f"\n{'='*50}")
        print(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"모델 파일: {Path(model_path).name}")
        print(f"{'='*50}")
        
        if isinstance(model, dict):
            return analyze_state_dict(model)
            
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n1. 기본 정보")
        print(f"{'-'*50}")
        print(f"총 파라미터 수: {total_params:,}")
        print(f"학습 가능한 파라미터 수: {trainable_params:,}")
        
        print("\n2. 레이어별 분석")
        print(f"{'-'*50}")
        
        layer_info = {}
        total_size_mb = 0
        
        for name, param in model.named_parameters():
            size_mb = param.element_size() * param.numel() / (1024 * 1024)
            total_size_mb += size_mb
            
            layer_type = name.split('.')[0]
            if layer_type not in layer_info:
                layer_info[layer_type] = {
                    'count': 0,
                    'params': 0,
                    'size_mb': 0
                }
            
            layer_info[layer_type]['count'] += 1
            layer_info[layer_type]['params'] += param.numel()
            layer_info[layer_type]['size_mb'] += size_mb
            
            print(f"\n레이어: {name}")
            print(f"Shape: {list(param.shape)}")
            print(f"파라미터 수: {param.numel():,}")
            print(f"크기: {size_mb:.2f} MB")
        
        print(f"\n{'='*50}")
        print("\n3. 레이어 타입별 요약")
        print(f"{'-'*50}")
        
        for layer_type, info in layer_info.items():
            print(f"\n{layer_type}:")
            print(f"레이어 수: {info['count']}")
            print(f"총 파라미터: {info['params']:,}")
            print(f"메모리 사용: {info['size_mb']:.2f} MB")
            
        print(f"\n총 모델 크기: {total_size_mb:.2f} MB")
        
        # 메모리 사용 정보 추가
        print("\n4. 메모리 사용 상세")
        print(f"{'-'*50}")
        print(f"파라미터 저장 공간: {total_size_mb:.2f} MB")
        print(f"추정 실행 메모리: {total_size_mb * 3:.2f} MB (파라미터 * 3 추정치)")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layer_info': layer_info,
            'total_size_mb': total_size_mb
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\n스택 트레이스:")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # 원래의 stdout으로 복구
        sys.stdout = sys.stdout.terminal

def analyze_state_dict(state_dict):
    """
    state_dict 형태의 모델 파라미터 분석
    """
    total_params = sum(param.numel() for param in state_dict.values())
    total_size_mb = sum(param.element_size() * param.numel() / (1024 * 1024) 
                       for param in state_dict.values())
    
    print("\n1. 기본 정보")
    print(f"{'-'*50}")
    print(f"총 파라미터 수: {total_params:,}")
    
    print("\n2. 레이어별 분석")
    print(f"{'-'*50}")
    
    for name, param in state_dict.items():
        size_mb = param.element_size() * param.numel() / (1024 * 1024)
        print(f"\n레이어: {name}")
        print(f"Shape: {list(param.shape)}")
        print(f"파라미터 수: {param.numel():,}")
        print(f"크기: {size_mb:.2f} MB")
    
    print(f"\n총 모델 크기: {total_size_mb:.2f} MB")
    
    return {
        'total_params': total_params,
        'total_size_mb': total_size_mb
    }

if __name__ == "__main__":
    # 원본 모델 파일 분석
    model_path = "model_optimized.ptl"  # 분석할 모델 파일 경로
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"model_analysis_{timestamp}.txt"
    
    print(f"분석을 시작합니다. 결과는 {log_file}에 저장됩니다.")
    model_info = analyze_model(model_path, log_file)
    print(f"분석이 완료되었습니다. 결과는 {log_file}에서 확인할 수 있습니다.")