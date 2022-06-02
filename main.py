import os
import random
import argparse
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch

from models import RNN, LSTM, AttentionalLSTM, CNN
from utils import make_dirs, load_data, data_loader, split_sequence_uni_step, split_sequence_multi_step, \
    inverse_transform_col
from utils import get_lr_scheduler, mean_percentage_error, mean_absolute_percentage_error, plot_pred_test

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    paths = [args.weights_path, args.plots_path]
    for path in paths:
        make_dirs(path)

    # 加载数据
    data = load_data(args.which_data)[args.feature]
    data = data.copy()
    print(data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    if args.multi_step:
        X, y = split_sequence_multi_step(data, args.seq_length, args.output_size)
        step = 'MultiStep'
    else:
        X, y = split_sequence_uni_step(data, args.seq_length)
        step = 'SingleStep'

    train_loader, val_loader, test_loader = data_loader(X, y, args.train_split, args.test_split, args.batch_size)

    train_losses, val_losses = list(), list()
    test_maes, test_mses, test_rmses, test_mapes, test_mpes = list(), list(), list(), list(), list()
    pred_tests, labels = list(), list()

    best_val_loss = 100
    best_val_improv = 0

    if args.model == 'cnn':
        model = CNN(args.seq_length, args.batch_size, args.output_size).to(device)
    elif args.model == 'rnn':
        model = RNN(args.input_size, args.hidden_size, args.num_layers, args.output_size).to(device)
    elif args.model == 'lstm':
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.bidirectional).to(
            device)
    elif args.model == 'attentional':
        model = AttentionalLSTM(args.input_size, args.qkv, args.hidden_size, args.num_layers, args.output_size,
                                args.bidirectional).to(device)
    else:
        raise NotImplementedError

    # 损失函数
    criterion = torch.nn.MSELoss()

    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_scheduler = get_lr_scheduler(args.lr_scheduler, optim)

    if args.mode == 'train':

        # 训练
        print("Training {} using {} started with total epoch of {}.".format(model.__class__.__name__, step,
                                                                            args.num_epochs))
        for epoch in range(args.num_epochs):
            for i, (data, label) in enumerate(train_loader):
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)

                pred = model(data)
                train_loss = criterion(pred, label)

                optim.zero_grad()
                train_loss.backward()
                optim.step()

                train_losses.append(train_loss.item())

            if (epoch + 1) % args.print_every == 0:
                print("Epoch [{}/{}]".format(epoch + 1, args.num_epochs))
                print("Train Loss {:.8f}".format(np.average(train_losses)))
            optim_scheduler.step()

            # 验证
            with torch.no_grad():
                for i, (data, label) in enumerate(val_loader):
                    data = data.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.float32)

                    pred_val = model(data)

                    val_loss = criterion(pred_val, label)

                    val_losses.append(val_loss.item())

            if (epoch + 1) % args.print_every == 0:
                print("Val Loss {:.8f}".format(np.average(val_losses)))

                # 保存模型
                curr_val_loss = np.average(val_losses)
                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)
                    torch.save(model.state_dict(), os.path.join(args.weights_path,
                                                                'BEST_{}_using_{}_Window_{}_Before_{}.pkl'.format(
                                                                    model.__class__.__name__, step,
                                                                    args.seq_length,
                                                                    args.output_size)))

                    print(epoch, "Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += 1
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

    elif args.mode == 'test':
        # 加载模型
        model.load_state_dict(
            torch.load(os.path.join(args.weights_path,
                                    'BEST_{}_using_{}_Window_{}_Before_{}.pkl'.format(model.__class__.__name__, step,
                                                                                      args.seq_length,
                                                                                      args.output_size))))
        print(model)
        err10 = []
        err20 = []
        err5 = []
        # 测试
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):
                # 数据准备
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)
                print(data.shape)
                pred_test = model(data)

                # 返归一化
                pred_test, label = pred_test.detach().cpu().numpy(), label.detach().cpu().numpy()
                label = inverse_transform_col(scaler, label, 0)
                pred_test = inverse_transform_col(scaler, pred_test, 0)
                if args.multi_step:
                    pred_test = pred_test[:,-1]
                    label = label[:,-1]
                    pred_tests += pred_test.tolist()
                    labels += label.tolist()
                else:
                    pred_tests += pred_test.tolist()
                    labels += label.tolist()

                # 误差
                test_mae = mean_absolute_error(label, pred_test)
                test_mse = mean_squared_error(label, pred_test, squared=True)
                test_rmse = mean_squared_error(label, pred_test, squared=False)
                test_mpe = mean_percentage_error(label, pred_test)
                test_mape = mean_absolute_percentage_error(label, pred_test)

                test_maes.append(test_mae.item())
                test_mses.append(test_mse.item())
                test_rmses.append(test_rmse.item())
                test_mpes.append(test_mpe.item())
                test_mapes.append(test_mape.item())
            err = []
            if args.multi_step:
                for i in range(len(pred_tests)):
                    err.append(abs(pred_tests[i] - labels[i]) / labels[i])
            else:
                for i in range(len(pred_tests)):
                    err.append(abs(pred_tests[i][0] - labels[i][0]) / labels[i][0])

            for i in range(len(pred_tests)):
                if err[i] <= 0.05:
                    err5.append(err[i])
                if err[i] <= 0.1:
                    err10.append(err[i])
                if err[i] <= 0.2:
                    err20.append(err[i])

            print("Test {} using {} and SeqLength {}".format(model.__class__.__name__, step, args.seq_length))
            print(" MAE : {:.4f}".format(np.average(test_maes)))
            print(" MSE : {:.4f}".format(np.average(test_mses)))
            print("RMSE : {:.4f}".format(np.average(test_rmses)))
            print(" MPE : {:.4f}".format(np.average(test_mpes)))
            print("MAPE : {:.4f}".format(np.average(test_mapes)))
            print(" ERR5 : {:.4f}".format(len(err5) / len(pred_tests)))
            print(" ERR10 : {:.4f}".format(len(err10) / len(pred_tests)))
            print(" ERR20 : {:.4f}".format(len(err20) / len(pred_tests)))

            # 绘图
            plot_pred_test(pred_tests[:], labels[:], args.plots_path, args.target, model,
                           step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7777, help='seed for reproducibility')
    parser.add_argument('--target', type=str, default='tplb', help='prediction target')
    parser.add_argument('--multi_step', type=bool, default=False, help='multi-step or not')
    parser.add_argument('--seq_length', type=int, default=8, help='window size')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--feature', type=list, default=['tplb', 'u10', 'v10', 'd2m', 't2m', 'sst', 'sp'],
                        help='use which features')

    parser.add_argument('--plot_full', type=bool, default=False, help='plot full graph or not')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    parser.add_argument('--model', type=str, default='attentional',
                        choices=['cnn', 'rnn', 'lstm', 'attentional'])
    parser.add_argument('--input_size', type=int, default=7, help='input_size')
    parser.add_argument('--hidden_size', type=int, default=10, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--output_size', type=int, default=1, help='output_size')
    parser.add_argument('--bidirectional', type=bool, default=False, help='use bidirectional or not')
    parser.add_argument('--qkv', type=int, default=7, help='dimension for query, key and value')

    parser.add_argument('--which_data', type=str, default='./data/data_all_mean_filter.csv', help='which data to use')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--plots_path', type=str, default='./results/plots/', help='plots path')

    parser.add_argument('--train_split', type=float, default=0.6, help='train_split')
    parser.add_argument('--test_split', type=float, default=0.5, help='test_split')

    parser.add_argument('--time_plot', type=int, default=500, help='time stamp for plotting')
    parser.add_argument('--num_epochs', type=int, default=1000, help='total epoch')
    parser.add_argument('--print_every', type=int, default=500, help='print statistics for every default epoch')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler',
                        choices=['step', 'plateau', 'cosine'])

    config = parser.parse_args()

    torch.cuda.empty_cache()
    main(config)
