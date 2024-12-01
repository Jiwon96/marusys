import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import traceback
from typing import Optional

def debug_convert_pt_to_ptl(pt_path: str, ptl_path: str) -> None:
    try:
        print("Step 1: Loading model...")
        # 모델 로드 방식 수정
        model = torch.jit.load(pt_path)
        print("Model loaded successfully")
        
        print("Step 2: Setting eval mode...")
        model.eval()
        
        print("Step 3: Testing model...")
        test_input = torch.randn(1, 1, 40, 101)
        with torch.no_grad():
            # 출력 타입 확인
            output = model(test_input)
            print(f"Model output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"Tuple contents: {[type(x) for x in output]}")
            print("Model test successful")
        
        print("Step 4: Starting mobile optimization...")
        # 모델 출력 타입 확인 후 처리
        if not isinstance(output, tuple):
            print("Warning: Model output is not a tuple, wrapping output...")
            original_forward = model.forward
            def wrapped_forward(*args, **kwargs):
                out = original_forward(*args, **kwargs)
                if not isinstance(out, tuple):
                    out = (out,)
                return out
            model.forward = wrapped_forward
        
        optimized = optimize_for_mobile(model)
        print("Mobile optimization successful")
        
        print("Step 5: Saving to PTL...")
        optimized._save_for_lite_interpreter(ptl_path)
        print("Successfully saved to PTL")
        
    except Exception as e:
        print(f"Failed at: {str(e)}")
        print("Detailed error:")
        traceback.print_exc()

def verify_model_output(model_path: str) -> None:
    """모델의 출력 형식을 검증하는 함수"""
    try:
        model = torch.jit.load(model_path)
        model.eval()
        
        test_input = torch.randn(1, 1, 40, 101)
        with torch.no_grad():
            output = model(test_input)
            
        print(f"\n모델 출력 검증 결과:")
        print(f"출력 타입: {type(output)}")
        if isinstance(output, tuple):
            print("튜플 내용:")
            for i, item in enumerate(output):
                print(f"  항목 {i}: {type(item)}")
        else:
            print(f"출력값: {output}")
            
    except Exception as e:
        print(f"모델 검증 중 오류 발생: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    model_path = "/home/ssafy/bcresnet-main/torchtest/model_scripted.pt"
    output_path = "model_mobile.ptl"
    
    print("모델 출력 형식 검증 중...")
    verify_model_output(model_path)
    
    print("\n모델 변환 시작...")
    debug_convert_pt_to_ptl(model_path, output_path)