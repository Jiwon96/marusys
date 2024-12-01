import torch
import json
from collections import OrderedDict
import os

def analyze_torchscript_model(model_path, output_path):
    """
    TorchScript 모델 파일을 로드하고 구조를 분석하여 JSON 파일로 저장합니다.
    
    Args:
        model_path (str): TorchScript 모델 파일 경로
        output_path (str): 결과를 저장할 JSON 파일 경로
    """
    try:
        # TorchScript 모델 로드
        model = torch.jit.load(model_path, map_location=torch.device('cpu'))
        
        def get_module_info(module, prefix=''):
            """재귀적으로 모듈 정보를 수집합니다."""
            module_info = OrderedDict()
            
            # 파라미터 분석
            for name, param in module.named_parameters(recurse=False):
                full_name = f"{prefix}.{name}" if prefix else name
                param_info = {
                    'shape': list(param.shape),
                    'num_parameters': param.numel(),
                    'dtype': str(param.dtype),
                    'requires_grad': bool(param.requires_grad)
                }
                
                # 파라미터 타입 추정
                if 'weight' in name:
                    if len(param.shape) == 4:
                        param_info['type'] = 'Convolution'
                    elif len(param.shape) == 2:
                        param_info['type'] = 'Linear'
                    else:
                        param_info['type'] = 'Other'
                elif 'bias' in name:
                    param_info['type'] = 'Bias'
                else:
                    param_info['type'] = 'Other'
                
                module_info[full_name] = param_info
            
            # 버퍼 분석
            for name, buf in module.named_buffers(recurse=False):
                full_name = f"{prefix}.{name}" if prefix else name
                buffer_info = {
                    'shape': list(buf.shape),
                    'num_parameters': buf.numel(),
                    'dtype': str(buf.dtype),
                    'type': 'Buffer'
                }
                module_info[full_name] = buffer_info
            
            # 하위 모듈 재귀적 분석
            for name, child in module.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                child_info = get_module_info(child, child_prefix)
                module_info.update(child_info)
            
            return module_info
        
        # 모델 구조 분석
        model_info = get_module_info(model)
        
        # 전체 모델 통계 계산
        total_params = sum(info['num_parameters'] for info in model_info.values())
        total_size = sum(info['num_parameters'] * {
            'torch.float32': 4,
            'torch.float64': 8,
            'torch.int32': 4,
            'torch.int64': 8
        }.get(info['dtype'], 4) for info in model_info.values())
        
        model_statistics = {
            'total_parameters': total_params,
            'num_layers': len(model_info),
            'model_size_mb': total_size / (1024 * 1024)
        }
        
        # 모델 코드 구조 (TorchScript 특화)
        try:
            model_code = model.code
            model_statistics['model_code'] = str(model_code)
        except:
            model_statistics['model_code'] = "코드 추출 불가"
        
        # 최종 결과 구성
        final_output = {
            'model_statistics': model_statistics,
            'layer_details': model_info
        }
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
            
        print(f"모델 구조가 성공적으로 분석되어 {output_path}에 저장되었습니다.")
        print(f"총 파라미터 수: {total_params:,}")
        print(f"총 레이어 수: {len(model_info)}")
        print(f"모델 크기: {model_statistics['model_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    model_path = "/home/ssafy/bcresnet-main/model_optimized.ptl"  # 실제 모델 경로로 변경하세요
    output_path = "model_structure.json"
    analyze_torchscript_model(model_path, output_path)