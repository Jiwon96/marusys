import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

def split_wav_files(input_dir, output_dir, train_size=0.8, valid_size=0.1, test_size=0.1, random_state=42):
    """
    WAV 파일들을 train, validation, test 세트로 분할하고 각각의 디렉토리에 복사합니다.
    
    Parameters:
    input_dir (str): WAV 파일들이 있는 입력 디렉토리 경로
    output_dir (str): 분할된 파일들이 저장될 출력 디렉토리 경로
    train_size (float): 학습 데이터 비율 (기본값: 0.8)
    valid_size (float): 검증 데이터 비율 (기본값: 0.1)
    test_size (float): 테스트 데이터 비율 (기본값: 0.1)
    random_state (int): 랜덤 시드 (기본값: 42)
    """
    # WAV 파일 리스트 가져오기
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    if not wav_files:
        raise Exception("WAV 파일을 찾을 수 없습니다.")
    
    # 먼저 train과 temp로 분할 (temp는 나중에 valid와 test로 나눕니다)
    train_files, temp_files = train_test_split(
        wav_files,
        train_size=train_size,
        random_state=random_state
    )
    
    # temp를 valid와 test로 분할
    # valid_size와 test_size를 temp 크기에 맞게 비율 조정
    valid_ratio = valid_size / (valid_size + test_size)
    valid_files, test_files = train_test_split(
        temp_files,
        train_size=valid_ratio,
        random_state=random_state
    )
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    test_dir = os.path.join(output_dir, 'test')
    
    for directory in [train_dir, valid_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 파일 복사 함수
    def copy_files(file_list, dest_dir):
        for file in file_list:
            src_path = os.path.join(input_dir, file)
            dst_path = os.path.join(dest_dir, file)
            shutil.copy2(src_path, dst_path)
    
    # 각 세트별로 파일 복사
    copy_files(train_files, train_dir)
    copy_files(valid_files, valid_dir)
    copy_files(test_files, test_dir)
    
    # 결과 출력
    print(f"전체 파일 수: {len(wav_files)}")
    print(f"학습 데이터: {len(train_files)} 파일 ({len(train_files)/len(wav_files)*100:.1f}%)")
    print(f"검증 데이터: {len(valid_files)} 파일 ({len(valid_files)/len(wav_files)*100:.1f}%)")
    print(f"테스트 데이터: {len(test_files)} 파일 ({len(test_files)/len(wav_files)*100:.1f}%)")
    
    return {
        'train_files': train_files,
        'valid_files': valid_files,
        'test_files': test_files
    }

# 사용 예시
if __name__ == "__main__":
    # 입력 디렉토리와 출력 디렉토리 설정
    input_directory = "/home/ssafy/bcresnet-main/S207Data/false"  # WAV 파일이 있는 디렉토리
    output_directory = "/home/ssafy/bcresnet-main/data/hey_ssafy"  # 분할된 데이터가 저장될 디렉토리
    
    # 데이터 분할 실행
    split_results = split_wav_files(
        input_directory,
        output_directory,
        train_size=0.8,
        valid_size=0.1,
        test_size=0.1
    )