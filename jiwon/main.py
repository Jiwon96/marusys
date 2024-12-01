# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from argparse import ArgumentParser
import shutil
from glob import glob
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from bcresnet import BCResNets
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--ver", default=1, help="google speech command set version 1 or 2", type=int
        )
        parser.add_argument(
            "--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        # 모델 저장 경로를 위한 인자 추가
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        print(self.device)

        self._load_data()
        self._load_model()

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """
        # train hyperparameters
        total_epoch = 20
        warmup_epoch = 5
        init_lr = 1e-2
        lr_lower_limit = 0
        best_acc = 0.0  # 최고 정확도 추적
        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9)
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        # train
        for epoch in range(total_epoch):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)

                loss = F.cross_entropy(outputs, labels) #0, 1 레이블 수정

                ##### score 체크용
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            train_loss = epoch_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)    

            # valid
            print("cur lr check ... %.4f" % lr)
            with torch.no_grad():
                self.model.eval()
                valid_acc, valid_loss = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                
                valid_accs.append(valid_acc)
                valid_losses.append(valid_loss)
                
                print("valid acc: %.3f %.3f" % (valid_acc, valid_loss))

                # 최고 정확도 모델 저장
                if valid_acc > best_acc:
                    best_acc = valid_acc
        print(f"Train losses: {train_losses}")
        print(f"Valid losses: {valid_losses}")
        print(f"Train accuracies: {train_accs}")
        print(f"Valid accuracies: {valid_accs}\n")
        self.draw_plot(train_losses, valid_losses, train_accs, valid_accs)
        test_acc, _ = self.Test(self.test_dataset, self.test_loader, augment=False)  # official testset
        print("test acc: %.3f" % (test_acc))
        print("End.")

    

    def Test(self, dataset, loader, augment):
        """
        Tests the model on a given dataset.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching the data.
            augment (bool): Flag indicating whether to use data augmentation during testing.

        Returns:
            float: The accuracy of the model on the given dataset.
        """

        print(f"Total test data: {len(dataset)}")
        print(f"Number of batches: {len(loader)}")
        

        true_count = 0.0
        val_loss = 0
        avg_val_loss=0
        num_testdata = float(len(dataset))

            # hey_ssafy 관련 메트릭을 위한 변수들 (augment=False 일 때만 사용)
        if not augment:
            hey_ssafy_idx = 2  # hey_ssafy 클래스의 인덱스
            all_predictions = []
            all_labels = []
            all_probabilities = []

        for inputs, labels in loader:
            #print("Labels:", labels)  # labels 출력
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)

            # Calculate loss
            loss = F.cross_entropy(outputs, labels)
            val_loss += loss.item()

            prediction = torch.argmax(outputs, dim=-1)
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()

            # augment가 False일 때만 confusion matrix를 위한 데이터 수집
            if not augment:
                # 각 배치의 결과를 리스트에 추가
                all_predictions.extend(prediction.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                probabilities = torch.softmax(outputs, dim=1)[:, hey_ssafy_idx].detach()
                all_probabilities.extend(probabilities.cpu().numpy())
            

        acc = true_count / num_testdata * 100.0  # percentage
        avg_val_loss = val_loss / len(loader)  # average loss per batch
        
        # augment가 False일 때만 hey_ssafy 메트릭 계산
        if not augment:
            # 모든 배치의 결과를 numpy 배열로 변환
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            all_probabilities = np.array(all_probabilities)
            
            # hey_ssafy 클래스에 대한 이진 레이블 생성
            hey_ssafy_true = (all_labels == hey_ssafy_idx)
            hey_ssafy_pred = (all_predictions == hey_ssafy_idx)
            
            # 전체 데이터에 대한 confusion matrix 계산
            tn = np.sum((~hey_ssafy_true) & (~hey_ssafy_pred))
            fp = np.sum((~hey_ssafy_true) & hey_ssafy_pred)
            fn = np.sum(hey_ssafy_true & (~hey_ssafy_pred))
            tp = np.sum(hey_ssafy_true & hey_ssafy_pred)
            
            print("\nData counts:")
            print(f"Total samples: {len(all_labels)}")
            print(f"hey_ssafy samples: {np.sum(hey_ssafy_true)}")
            
            # 메트릭 계산
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # ROC curve와 AUC 계산
            fpr, tpr, _ = roc_curve(hey_ssafy_true, all_probabilities)
            roc_auc = auc(fpr, tpr)
            
            # 결과 출력
            print("\nTest Set Metrics for hey_ssafy class:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            
            # Confusion Matrix 시각화
            plt.figure(figsize=(8, 6))
            cm = np.array([[tn, fp], [fn, tp]])
            sns.heatmap(cm, annot=True, fmt='g', 
                    xticklabels=['Not hey_ssafy', 'hey_ssafy'],
                    yticklabels=['Not hey_ssafy', 'hey_ssafy'])
            plt.title('Test Set Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # ROC Curve 시각화
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for hey_ssafy')
            plt.legend(loc="lower right")
            plt.savefig('test_roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 메트릭을 CSV 파일로 저장
            metrics = {
                'accuracy': acc,
                'loss': avg_val_loss,
                'hey_ssafy_precision': precision,
                'hey_ssafy_recall': recall,
                'hey_ssafy_f1': f1,
                'hey_ssafy_roc_auc': roc_auc
            }
            pd.DataFrame([metrics]).to_csv('test_metrics.csv', index=False)
        
        return acc, avg_val_loss

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """
        print("Check google speech commands dataset v1 or v2 ...")
        if not os.path.isdir("./data"):
            os.mkdir("./data")
        base_dir = "./data/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        url_test = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
            url = url.replace("v0.01", "v0.02")
            url_test = url_test.replace("v0.01", "v0.02")
        elif self.ver == 3:
            base_dir = './data/hey_ssafy'
        elif self.ver == 4:
            base_dir = './data/hey_ssafy2'
        
        if self.ver ==3:
            test_dir = base_dir+"_test_set"
        elif self.ver==4:
            test_dir = base_dir[:-1]+"_test_set"
        else:
            test_dir = base_dir.replace("commands", "commands_test_set") # 시그모이드
        

        if self.download:
            old_dirs = glob(base_dir.replace("commands_", "commands_*"))
            for old_dir in old_dirs:
                shutil.rmtree(old_dir)
            os.mkdir(test_dir)
            DownloadDataset(test_dir, url_test)
            os.mkdir(base_dir)
            DownloadDataset(base_dir, url)
            SplitDataset(base_dir)
            print("Done...")

        train_dir = "%s/train_12class" % base_dir
        valid_dir = "%s/valid_12class" % base_dir
        noise_dir = "%s/_background_noise_" % base_dir

        transform = transforms.Compose([Padding()])
        self.train_dataset = SpeechCommand(train_dir, self.ver, transform=transform)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=100, shuffle=True, num_workers=0, drop_last=False
        )
        print('test_dir', test_dir)
        self.valid_dataset = SpeechCommand(valid_dir, self.ver, transform=transform)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=100, num_workers=0)
        self.test_dataset = SpeechCommand(test_dir, self.ver, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, num_workers=0)

        print(
            "check num of data train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        # Define preprocessors
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device)

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        print("model: BC-ResNet-%.1f on data v0.0%d" % (self.tau, self.ver))
        self.model = BCResNets(int(self.tau * 8)).to(self.device)

    def save_for_mobile(self):
            """
            Save the model in a format optimized for mobile.
            """
            # 학습 완료 후 모델을 CPU로 이동
            self.model.eval()
            self.model.to('cpu')

            # CPU에서의 입력 샘플 생성
            input_sample = torch.randn(1,1, 40, 201) # > 1,1, 40, 101에서 수정

            # TorchScript로 변환
            traced_model = torch.jit.trace(self.model, input_sample, strict=True, check_trace=True)

            with torch.no_grad():
                output= traced_model(input_sample)
                print(f"Input shape: {input_sample.shape}")
                print(f"Output shape: {output.shape}")

            traced_model.to('cpu')  # CPU로 변환된 모델 저장
            traced_model.save("model_scripted.pt")  # 일반 TorchScript 모델 저장


            # 모바일 최적화 적용
            optimized_model = optimize_for_mobile(traced_model)
            optimized_model._save_for_lite_interpreter("model_optimized.ptl")  # 모바일 최적화 모델 저장

            print("\nOptimized model saved as model_optimized.ptl")

    def draw_plot(self, train_losses, val_losses, train_accs, val_accs):
        """
        학습 과정의 Loss와 Accuracy를 시각화하는 함수

        Args:
            train_losses (list): 에폭별 학습 Loss 값들을 담은 리스트
            val_losses (list): 에폭별 검증 Loss 값들을 담은 리스트 
            train_accs (list): 에폭별 학습 Accuracy 값들을 담은 리스트
            val_accs (list): 에폭별 검증 Accuracy 값들을 담은 리스트

        Returns:
            None. 그래프를 'training_metrics.png' 파일로 저장함
        """
        epochs = list(range(len(train_losses)))  # 실제 에폭 수에 맞게 x축 생성
        
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-o', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)  # y축 범위 설정
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b-o', label='Train Acc')
        plt.plot(epochs, val_accs, 'r-o', label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)  # accuracy는 0-100 사이
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    _trainer = Trainer()
    _trainer()

    feature = _trainer.preprocess_test.feature
    feature.device = 'cpu'  # device 속성 변경
    feature.mel = feature.mel.to('cpu')
    feature = feature.to('cpu')
    
    input_sample = torch.randn(1, 32000)
    
    try:
        print("모델이 성공적으로 저장되었습니다.")
        traced_model = torch.jit.trace(feature, input_sample)
        traced_model.save("logmel_scripted.pt")
        
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized_model = optimize_for_mobile(traced_model)
        optimized_model._save_for_lite_interpreter("logmel_optimized.ptl")
        print("Optimized model saved as logmel_optimized.ptl")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

    _trainer.save_for_mobile()