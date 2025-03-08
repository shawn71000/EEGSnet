import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.clock_driven import functional
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from Model.modelDE import *
from Feature import *
from config import get_config
from utils import *
import numpy as np

# ====== cross-subject ======

_seed_ = 2024
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


def main():
    args = get_config()

    # divide
    # ==============================================
    datasetList = ['Fatigue', 'SEED', 'Cognitive']

    # ==Dataset
    for d, dataset in enumerate(datasetList):
        args.dataset = dataset
        print('dataset:', args.dataset)

        # save print
        path = os.path.abspath(os.path.dirname(__file__)) + '/logs/Prints/' + args.dataset
        timeShow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        sys.stdout = Logger(filename=path + '/' + timeShow + args.dataset + '_Print.txt')
        labels = []

        if args.dataset == 'Fatigue':
            # model load

            # data load, one subject used to test
            sub_list = ['ARCALE', 'ANZALE', 'BORGIA', 'CULLEO', 'CILRAM', 'CALGIO', 'DESTER', 'DIFANT', 'GNATN', 'MESMAR', 'MARFRA', 'SCAEMI', 'SALSTE', 'VALNIC', 'VALPAO']
            data_dir_list = ['Fatigue/']

            # param load
            build_DE_eeg_dataset = BuildDEFatigueTool.build_DE_eeg_dataset
            subject_cross_data_split = BuildDEFatigueTool.subject_cross_data_split
            path_dataset = 'Datasets-Origin/Fatigue-DE/'
            sampleNum = 1400
            num_feats = 1647
            num_nodes = 61 # channels
            num_freq = 27
            args.input_channels = 5
            args.num_classes = 2 # classes of label
            args.channels_FC1 = 4
            args.channels_FC2 = 5
            args.input_shape1 = 8
            args.input_shape2 = 9
            args.channels_Alex1 = 1
            args.channels_Alex2 = 1
            args.avgpool_Alex = 2
            labels = ['Fatigue', 'Wakefulness']

        elif args.dataset == 'SEED':

            sub_list = ['1', '2', '3', '4', '5', '6', '7', '8',
                           '9', '10', '11', '12', '13', '14', '15']
            data_dir_list = ['SEED1/', 'SEED2/', 'SEED3/']  #

            build_DE_eeg_dataset = BuildDESEEDTool.build_DE_eeg_dataset
            subject_cross_data_split = BuildDESEEDTool.subject_cross_data_split
            path_dataset = 'Datasets-Origin/SEED-DE/'
            sampleNum = 842
            num_feats = 2604
            num_nodes = 62
            num_freq = 42
            args.input_channels = 5
            args.num_classes = 3 # classes of label
            args.channels_FC1 = 4
            args.channels_FC2 = 5
            args.input_shape1 = 8
            args.input_shape2 = 9
            args.channels_Alex1 = 1
            args.channels_Alex2 = 1
            args.avgpool_Alex = 2
            labels = ['Positive', 'Neutral', 'Negative']

        elif args.dataset == 'Cognitive':   #ADMH

            sub_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
            data_dir_list = ['Cog_happiness/', 'Cog_neutral/', 'Cog_sadness/']
            build_DE_eeg_dataset = BuildDECogTool.build_DE_eeg_dataset
            subject_cross_data_split = BuildDECogTool.subject_cross_data_split
            path_dataset = 'Datasets-Origin/AD-MCI-HC/'
            num_nodes = 32
            # num_freq = 0.5-30Hz
            args.input_channels = 4
            args.num_classes = 3 # classes of label
            args.channels_FC1 = 1
            args.channels_FC2 = 3
            args.input_shape1 = 7
            args.input_shape2 = 5
            args.channels_Alex1 = 1
            args.channels_Alex2 = 1
            args.avgpool_Alex = 1
            labels = ['AD', 'MCI', 'HC']

        else:
            print('error!')
            os._exit(0)

        avg_All = AverageMeter()  # all datasets avg acc
        avg_All_F1 = AverageMeter()  # all datasets avg acc
        avg_All_Recall = AverageMeter()
        avg_All_Pre = AverageMeter()
        avg_time = AverageMeter()  # epoch time
        avg_All_F1_array = []
        avg_All_array = []
        best_acc_array_Recall = []
        best_acc_array_Pre = []

        # ==Session
        for k in range(len(data_dir_list)):
            folder_path = './data_set/' + data_dir_list[k]
            print('======folder_path:', folder_path)
            Num_target = 0  # number of target
            # .mat->npy
            if args.dataset == 'SEED':
                origin_path = path_dataset + str(k + 1) + '/'
            else:
                origin_path = path_dataset + '/'

            # DE
            feature_vector_dict, label_dict = build_DE_eeg_dataset(folder_path, origin_path, dis=6, map_size=9)

            # print_best_acc_array
            best_acc_array = []
            avg_best_acc = AverageMeter()
            best_acc_array_F1 = []
            avg_best_F1 = AverageMeter()
            flod_Recall = AverageMeter()
            flod_Pre = AverageMeter()
            flod_Recall_array = []
            flod_Pre_array = []

            # ==Leave-one-subject-out LOSO
            for m, target_sample in enumerate(sub_list):
                Num_target += 1
                test_subjects = [target_sample]
                print('target_sample: ', test_subjects)
                train_subjects = []
                for _, source in enumerate(sub_list):
                    if test_subjects[0] == source:
                        continue
                    train_subjects.append(source)

                train_feature, train_label, test_feature, test_label = subject_cross_data_split(feature_vector_dict, label_dict, test_subjects, args.dataset)
                train_set = EEGDataset_DE(train_feature, train_label)
                test_set = EEGDataset_DE(test_feature, test_label)


                train_data_loader = DataLoader(
                    dataset=train_set,
                    batch_size=args.b,
                    shuffle=True,
                    num_workers=args.j,
                    drop_last=True,
                    pin_memory=False)

                test_data_loader = DataLoader(
                    dataset=test_set,
                    batch_size=args.b,
                    shuffle=True,
                    num_workers=args.j,
                    drop_last=True,
                    pin_memory=False)

                # model
                net = S_TCNTauNet(input_channels=args.input_channels, channels=args.channels, arg=args)

                print(net)
                print(args)
                net.to(args.device)

                optimizer = None
                if args.opt == 'SGD':
                    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
                elif args.opt == 'Adam':
                    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
                else:
                    raise NotImplementedError(args.opt)

                lr_scheduler = None
                if args.lr_scheduler == 'StepLR':
                    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                                                   gamma=args.gamma)
                elif args.lr_scheduler == 'CosALR':
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
                else:
                    raise NotImplementedError(args.lr_scheduler)

                scaler = None
                if args.amp:
                    scaler = amp.GradScaler()

                start_epoch = 0
                max_test_acc = 0
                best_F1 = 0
                best_Recall = 0
                best_Pre = 0

                out_dir = os.path.join(args.out_dir + args.dataset,
                                       f'cross_{args.dataset}{k+1}_t{target_sample}_b{args.b}_c{args.channels}_{args.opt}_lr{args.lr}_')
                if args.lr_scheduler == 'CosALR':
                    out_dir += f'CosALR_{args.T_max}'
                elif args.lr_scheduler == 'StepLR':
                    out_dir += f'StepLR_{args.step_size}_{args.gamma}'
                else:
                    raise NotImplementedError(args.lr_scheduler)

                if args.amp:
                    out_dir += '_amp'
                if args.cupy:
                    out_dir += '_cupy'

                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                    print(f'Mkdir {out_dir}.')

                with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
                    args_txt.write(str(args))

                writer = SummaryWriter(os.path.join(out_dir, 'dvsg_logs'), purge_step=start_epoch)

                for epoch in range(start_epoch, args.epochs):
                    start_time = time.time()
                    net.train()
                    train_loss = 0
                    train_acc = 0
                    train_samples = 0
                    for frame, label in train_data_loader:  # 训练
                        optimizer.zero_grad()
                        frame = frame.float().to(args.device)
                        label = label.to(args.device)
                        label_onehot = F.one_hot(label, args.num_classes).float()

                        if args.amp:
                            with amp.autocast():
                                out_fr = net(frame)
                                loss = F.mse_loss(out_fr, label_onehot)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            out_fr = net(frame)
                            loss = F.mse_loss(out_fr, label_onehot)
                            loss.backward()
                            optimizer.step()

                        train_samples += label.numel()
                        train_loss += loss.item() * label.numel()
                        train_acc += (out_fr.argmax(1) == label).float().sum().item()

                        functional.reset_net(net)
                    train_loss /= train_samples
                    train_acc /= train_samples

                    writer.add_scalar('train_loss', train_loss, epoch)
                    writer.add_scalar('train_acc', train_acc, epoch)
                    lr_scheduler.step()

                    net.eval()
                    test_loss = 0
                    test_acc = 0
                    test_samples = 0
                    Acc = AverageMeter()
                    F1 = AverageMeter()
                    Recall = AverageMeter()
                    Pre = AverageMeter()
                    score_list = []     # 存储预测得分
                    label_list = []     # 存储真实标签
                    confusion = ConfusionMatrix(num_classes=args.num_classes, args=args)
                    with torch.no_grad():
                        for frame, label in test_data_loader:
                            frame = frame.float().to(args.device)
                            label = label.to(args.device)
                            label_onehot = F.one_hot(label, args.num_classes).float()
                            out_fr = net(frame)
                            loss = F.mse_loss(out_fr, label_onehot)

                            test_samples += label.numel()
                            test_loss += loss.item() * label.numel()
                            test_acc += (out_fr.argmax(1) == label).float().sum().item()
                            confusion.update(out_fr.argmax(1).cpu(), label.cpu())
                            functional.reset_net(net)
                            score_list.extend(out_fr.detach().cpu().numpy())
                            label_list.extend(label.cpu().numpy())

                        Accuracy, F1score, Recall_t, Pre_t = confusion.summary()
                        Acc.update(Accuracy)
                        F1.update(F1score)
                        Recall.update(Recall_t)
                        Pre.update(Pre_t)

                        score_array = np.array(score_list)
                        # 将label转换成onehot形式
                        label_tensor = torch.tensor(label_list)
                        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
                        label_onehot = torch.zeros(label_tensor.shape[0], args.num_classes)
                        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
                        label_onehot = np.array(label_onehot)

                    test_loss /= test_samples
                    test_acc /= test_samples

                    writer.add_scalar('test_loss', test_loss, epoch)
                    writer.add_scalar('test_acc', Acc.avg, epoch)

                    save_max = False
                    if Acc.avg > max_test_acc:
                        max_test_acc = Acc.avg
                        save_max = True
                        best_F1 = F1.avg
                        best_Recall = Recall.avg
                        best_Pre = Pre.avg
                        confusion.plot(labels, out_dir, m)
                        confusion.paint_ROC(label_onehot, score_array, out_dir, m)

                    checkpoint = {
                        'net': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'max_test_acc': max_test_acc,
                        'max_test_F1': best_F1,
                        'max_test_Recall': best_Recall,
                        'max_test_Pre': best_Pre
                    }

                    if save_max:
                        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

                    torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

                    print(
                        f'target_sample={test_subjects}, epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={Acc.avg}, max_test_acc={max_test_acc}, F1={best_F1}, total_time={time.time() - start_time}')

                    avg_time.update(time.time() - start_time)

                best_acc_array.append(max_test_acc)
                avg_best_acc.update(max_test_acc)
                best_acc_array_F1.append(best_F1)
                avg_best_F1.update(best_F1)
                flod_Recall.update(best_Recall)
                flod_Recall_array.append(best_Recall)
                flod_Pre.update(best_Pre)
                flod_Pre_array.append(best_Pre)
                print(f'dataset={folder_path}, avg_best_acc={avg_best_acc.avg}, best_arr={best_acc_array}, avg_best_F1={avg_best_F1.avg}, best_arr={best_acc_array_F1}')

            avg_All.update(avg_best_acc.avg)
            avg_All_array.append(avg_best_acc.avg)
            avg_All_F1.update(avg_best_F1.avg)
            avg_All_F1_array.append(avg_best_F1.avg)
            best_acc_array_Recall.append(flod_Recall.avg)
            avg_All_Recall.update(flod_Recall.avg)
            best_acc_array_Pre.append(flod_Pre.avg)
            avg_All_Pre.update(flod_Pre.avg)
            print(f'avg_all={avg_All.avg}, best_arr={avg_All_array}, avg_all_F1={avg_All_F1.avg}, best_arr_F1={avg_All_F1_array}, best_arr_Recall={best_acc_array_Recall}, best_arr_Pre={best_acc_array_Pre}, avg_time={avg_time.avg}')


if __name__ == '__main__':
    main()
