import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn as nn
import torchvision.models as models

from data_loader import BasicDataset

parser = argparse.ArgumentParser(description='PyTorch DARC fine-tuning')
parser.add_argument('--data', metavar='DIR', default="./test_data",
                    help='path to dataset')
parser.add_argument('--model', default="./checkpoint/checkpoints.pt",
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-batch-size', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2)')
parser.add_argument('--seed', default=31, type=int,
                    help='seed for initializing training.')

args = parser.parse_args()

# 定义类别及其对应的索引
classes = {0: 'ADI', 1: 'BACK', 2: 'DEB', 3: 'LYM', 4: 'MUC', 5: 'MUS', 6: 'NORM', 7: 'STR', 8: 'TUM'}
class_to_idx = {v: k for k, v in classes.items()}


def test(test_loader, model):
    target_pred = dict()
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for input_tensor, image_filenames in test_loader:
            input_var = input_tensor.cuda()

            output = model(input_var)
            _, pred = torch.max(output.data, 1)

            for i, image_filename in enumerate(image_filenames):
                # 提取真实标签
                true_label_str = image_filename.split('-')[0]
                true_label = class_to_idx.get(true_label_str, None)

                if true_label is None:
                    print(f"未知的类别标签: {true_label_str} in file {image_filename}")
                    continue  # 跳过未知类别

                pred_label = pred[i].item()

                # 统计总体正确数和总数
                if pred_label == true_label:
                    correct += 1
                total += 1

                predicted_class = classes[pred_label]
                target_pred[image_filename] = predicted_class

                # 输出预测过程信息
                print(f"正在预测{image_filename}文件，预测结果为{predicted_class}。")

    # 计算总体准确率
    overall_accuracy = 100.0 * correct / total if total > 0 else 0.0

    return target_pred, overall_accuracy, total, correct


def load_model(path):
    model = models.resnet18(pretrained=True)
    num_fc = model.fc.in_features
    model.fc = nn.Linear(num_fc, 9)
    # print(model)
    if os.path.isfile(path):
        print(f"=> 正在加载检查点 '{path}'")
        checkpoint_dict = torch.load(path, map_location='cpu')
        try:
            a, b = model.load_state_dict(checkpoint_dict)
            print(a, b)
            print("模型加载成功。")
        except RuntimeError as e:
            print(f"加载模型时出错: {e}")
            return None
    else:
        print(f"=> 在 '{path}' 未找到检查点文件。")
        return None
    return model


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = load_model(args.model)
    model.eval()

    if model is None:
        print("模型加载失败，请检查模型路径和模型文件。")
        return

    model.cuda()
    cudnn.benchmark = True

    test_dataset = BasicDataset(args.data)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=int(args.batch_size),
                                              shuffle=False,
                                              num_workers=args.workers,
                                              drop_last=False)

    pred, overall_accuracy, total, correct = test(test_loader, model)

    with open('./result.txt', 'w') as file:
        for key in pred:
            file.write(f"{key} : {pred[key]}\n")

        file.write(f"\n总共预测{total}个数据，预测准确率为{overall_accuracy:.2f}%\n")

    print(f"总共预测{total}个数据，预测准确率为{overall_accuracy:.2f}%.")


if __name__ == '__main__':
    main()