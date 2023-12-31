import json
import sys
import random
import numpy as np
import torch.optim as optim
import argparse
import copy
import datetime
from utils import *
from loguru import logger
import torch
logger.remove()
# 为不同的日志级别添加handler并设置颜色
logger.add(sys.stdout, format="<green>{time}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="DEBUG", colorize=True, filter=lambda record: record["level"].name == "DEBUG")
logger.add(sys.stdout, format="<yellow>{time}</yellow> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO", colorize=True, filter=lambda record: record["level"].name == "INFO")
logger.add(sys.stdout, format="<blue>{time}</blue> | <level>{level: <8}</level> | <level>{message}</level>", level="WARNING", colorize=True, filter=lambda record: record["level"].name == "WARNING")
logger.add(sys.stdout, format="<red>{time}</red> | <level>{level: <8}</level> | <level>{message}</level>", level="ERROR", colorize=True, filter=lambda record: record["level"].name == "ERROR")
logger.add(sys.stdout, format="<magenta>{time}</magenta> | <level>{level: <8}</level> | <level>{message}</level>", level="CRITICAL", colorize=True, filter=lambda record: record["level"].name == "CRITICAL")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple-cnn-mnist', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=20, help='number of workers in a distributed cluster')
    parser.add_argument('--n_malicious_parties', type=int, default=6)
    parser.add_argument('--alg', type=str, default='basalt',
                        help='communication strategy: fedavg/fedprox/basalt/Median/trimmed_mean/krum')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--beta', type=float, default=2,
                        help='The parameter for penalty  constant')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=100, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')

    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--attack', type=str, default="label_flipping", help='attack methods:label_flipping,sign_flipping,Gaussian_attack')
    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net = torch.nn.DataParallel(net,).to(device)
    logger.info(f"Training network: {net}")
    logger.info(f"training dataset size: {(train_dataloader)}")
    logger.info(f"testing dataset size:{test_dataloader}")
    logger.info(f"dataset:{args.dataset}")
    optimizer = None
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    cnt = 0
    if net_id in [i for i in range(0, args.n_parties-args.n_malicious_parties)]:
        for epoch in range(epochs):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                _, _, out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            if epoch % 10 == 0:
                train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
                test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device)

                logger.info('>> Training accuracy: %f' % train_acc)
                logger.info('>> Test accuracy: %f' % test_acc)


    elif net_id in [i for i in range(args.n_parties-args.n_malicious_parties,args.n_parties)]:
        for epoch in range(epochs):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                target=torch.ones_like(target)
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                _, _, out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

            if epoch % 10 == 0:
                train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
                test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device)

                logger.info('>> Training accuracy: %f' % train_acc)
                logger.info('>> Test accuracy: %f' % test_acc)

    elif net_id in [i for i in range(args.n_parties-args.n_malicious_parties,args.n_parties)]:
        for epoch in range(epochs):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()

                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                _, _, out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

            if epoch % 10 == 0:
                train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
                test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device)

                logger.info('>> Training accuracy: %f' % train_acc)
                logger.info('>> Test accuracy: %f' % test_acc)
        temp_vector = torch.nn.utils.parameters_to_vector(net.parameters()) * (-4)
        # temp_vector=temp_vector.tolist()
        torch.nn.utils.vector_to_parameters(temp_vector, net.parameters())
    elif args.attack == "Gaussian_attack":
        local_model_vector = torch.nn.utils.parameters_to_vector(net.parameters())
        local_model_vector = torch.randn(local_model_vector.shape[0])
        torch.nn.utils.vector_to_parameters(local_model_vector, net.parameters())

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')
    return train_acc, test_acc



def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    net = nn.DataParallel(net).to(device)

    logger.info('client iD %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    optimizer = None
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    global_weight_collector = list(global_net.to(device).parameters())
    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg
            loss.backward()
            optimizer.step()
            cnt += 1
            epoch_loss_collector.append(loss.item())
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    return train_acc, test_acc

def train_net_fedcon(net_id, net, previous_nets, macilious_prev_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args,
                      round, device="cpu"):
    net = nn.DataParallel(net).to(device)
    logger.info(f"Training round: {round}, Client Id:{net_id},local training data size:{len(train_dataloader)},testing data size:{len(test_dataloader)}")
    # model-level penalty lass
    triple_loss = torch.nn.TripletMarginLoss(margin=args.alpha, p=2).to(device)
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    for previous_net in previous_nets:
        previous_net.to(device)
    for macilious_previous_net in macilious_prev_model:
        macilious_previous_net.to(device)
    cnt = 0
    if net_id in [i for i in range(0,args.n_parties-args.n_malicious_parties)]:
        for epoch in range(epochs):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()
                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()
                _, _, out = net(x)
                for previous_net in previous_nets:
                    previous_net.cuda()
                loss1 = criterion(out, target)
                anchor = torch.nn.utils.parameters_to_vector(net.parameters()).cuda().detach()
                positive = torch.nn.utils.parameters_to_vector(previous_net.parameters()).cuda()
                negative= torch.nn.utils.parameters_to_vector(macilious_previous_net.parameters()).cuda().detach()
                loss2 = triple_loss(anchor,positive,negative)
                loss = loss1+loss2
                loss.backward()
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))
        train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info(' ** Training complete **')
        return train_acc, test_acc
    elif args.attack=="label_flipping":
        for epoch in range(epochs):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                target=torch.ones_like(target)
                x, target = x.cuda(), target.cuda()
                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()
                _, pro1, out = net(x)
                for previous_net in previous_nets:
                    previous_net.cuda()
                loss = loss1 = criterion(out, target)
                loss.backward()
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            logger.info('Epoch: %d Loss: %f Loss1: %f' % (epoch, epoch_loss, epoch_loss1))
        train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
        logger.info(' ** Training complete **')
        return train_acc, test_acc
    elif args.attack == "sign_flipping":
        for epoch in range(epochs):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()
                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()
                _, _, out = net(x)
                loss = loss1 = criterion(out, target)
                loss.backward()
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
        temp_vector=torch.nn.utils.parameters_to_vector(net.parameters())*(-4)
        torch.nn.utils.vector_to_parameters(temp_vector,net.parameters())
        train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info(f"Training accuracy:{train_acc}")
        logger.info(f"Test accuracy: {test_acc}")
        logger.info(' ** Training complete **')
        return train_acc, test_acc
    elif args.attack == "Gaussian_attack":
        local_model_vector = torch.nn.utils.parameters_to_vector(net.parameters())
        local_model_vector = torch.randn(local_model_vector.shape[0])*(4)
        torch.nn.utils.vector_to_parameters(local_model_vector, net.parameters())
        net.cuda()
        train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info('>> Training accuracy: %f' % train_acc)
        logger.info('>> Test accuracy: %f' % test_acc)
        logger.info(' ** Training complete **')
        return train_acc, test_acc


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None,macilious_model_pool=None,server_c = None, clients_c = None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        if args.alg == 'fedavg':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'basalt':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            macilious_prev_model=macilious_model_pool
            trainacc, testacc = train_net_fedcon(net_id, net, prev_models,macilious_prev_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args, round, device=device)
            logger.info(f"cleint id: {net_id},train accuracy: {trainacc*100}%  test accuracy: {testacc*100}%")
        elif args.alg == 'local_training':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        elif args.alg == 'Median':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        elif args.alg == 'trimmed_mean':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        elif args.alg == 'krum':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to(device)
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to(device)
    return nets


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_dir = "logs"
    if not os.path.isdir(log_dir):
        os.makedirs("logs")
    log_path = log_dir+"/"+args.log_file_name + '.log'
    logger.add(log_path)

    logger.info(f"training device:{device}")

    #设置随机数种子

    seed = args.init_seed
    logger.info(f"random  seed: {seed}")
    np.random.seed(seed)#设置numpy的随机数种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


    #划分数据
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, alpha=args.alpha)



    #选择进行训练的客户端
    n_party_per_round = int(args.n_parties * args.sample_fraction)#n_party_per_round是每轮进行训练的客户端数量
    party_list = [i for i in range(args.n_parties)]#n_parties是客户端总数量
    party_list_rounds = []#记录每一轮参与训练的客户端的index
    if n_party_per_round != args.n_parties:#如果每一轮训练过程中没选择全部用户的话
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:#如果每一轮训练过程中选择全部用户的话
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    #准备数据,将数据转化为自己设计的dataset类
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)
    train_dl=None
    data_size = len(test_ds_global)
    #准备模型
    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device=device)
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device=device)
    global_model = global_models[0]
    #7.23 by LHS
    macilious_global_models, macilious_global_model_meta_data, macilious_global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    macilious_global_model = macilious_global_models[0]


    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    if args.alg == 'basalt':
        old_nets_pool = []
        macilious_old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():#过去的模型不进行更新
                        param.requires_grad = False
        #test
        old_nets_pool.append(old_nets)
        macilious_old_nets_pool.append(macilious_global_model)

        for round in range(n_comm_rounds):
            party_list_this_round = party_list_rounds[round]
            global_model.eval()
            for param in global_model.parameters():#globalmodel不进行更新
                param.requires_grad = False
            global_w = global_model.state_dict()
            #7.23 by LHS
            macilious_global_model.eval()
            for param in macilious_global_model.parameters():  # maciliousmodel不进行更新
                param.requires_grad = False
            macilious_global_w = macilious_global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():#全局模型下发至本地
                net.load_state_dict(global_w)

            #用户利用下发下来的全局模型进行本地训练
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool,macilious_model_pool=macilious_old_nets_pool,round=round, device=device)



            #计算聚合权重
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            #拜占庭用户检测
            matrix =[]
            for net_id, net in enumerate(nets_this_round.values()):
                vec = torch.nn.utils.parameters_to_vector(net.parameters()).tolist()
                matrix.append(vec)

            matrix = np.array(matrix)
            benign_sample, y_pred = umap_kmeans(matrix, party_list_this_round)
            logger.info(f"malicious prediction label: {y_pred}")
            logger.info(f"benign_client_id:{benign_sample}")
            logger.info(f"malicious_client_id:{set(range(args.n_parties))-set(benign_sample)}")

            #重新计算权重(良性用户权重)
            fed_avg_freqs=[]
            for i in range(len(party_list_this_round)):
                if i in benign_sample:
                    fed_avg_freqs.append(1/len(benign_sample))
                else:
                    fed_avg_freqs.append(0)
            # 重新计算权重(恶意用户权重)
            macilious_avg_freqs = []
            for i in range(len(party_list_this_round)):
                if i not in benign_sample:
                    macilious_avg_freqs.append(1 / (args.n_parties-len(benign_sample)))
                else:
                    macilious_avg_freqs.append(0)

            #加权聚合(良性用户)
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]
            global_model.load_state_dict(global_w)
            # 加权聚合(恶意用户)
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        macilious_global_w[key] = net_para[key] * macilious_avg_freqs[net_id]
                else:
                    for key in net_para:
                        macilious_global_w[key] += net_para[key] * macilious_avg_freqs[net_id]

            macilious_global_model.load_state_dict(macilious_global_w)
            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            logger.info('>> Basalt Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Basalt Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Basalt Global Model Train loss: %f' % train_loss)
            macilious_old_nets_pool.append(macilious_global_model)

            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')


    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]#这一轮参与训练的所有客户

            global_w = global_model.state_dict()
            if args.server_momentum:#滑动平均
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}#nets_this_round本轮中参与训练的所有用户的模型
            for net in nets_this_round.values():
                net.load_state_dict(global_w)#将本地模型替换为全局模型

            #客户利用替换后的全局模型进行本地训练
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

            #计算每个用户聚合时的权重
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            #联邦平均加权聚合
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            # 如果使用滑动平均
            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            logger.info('>> fedavg Global Model Train accuracy: %f' % train_acc)
            logger.info('>> fedavg Global Model Test accuracy: %f' % test_acc)
            logger.info('>> fedavg Global Model Train loss: %f' % train_loss)
            print(" fedavg Global Model Train accuracy" + str(train_acc))
            print("fedavg Global Model Test accuracy" + str(test_acc))

    elif args.alg == 'fedprox':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to(device)
            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)
            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')
    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device=device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')
    elif args.alg == 'Median':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]  # 这一轮参与训练的所有客户
            global_w = global_model.state_dict()
            if args.server_momentum:  # 滑动平均
                old_w = copy.deepcopy(global_model.state_dict())
            nets_this_round = {k: nets[k] for k in party_list_this_round}  # nets_this_round本轮中参与训练的所有用户的模型
            for net in nets_this_round.values():
                # net_vector = torch.nn.utils.parameters_to_vector(net.parameters())
                net.load_state_dict(global_w)  # 将本地模型替换为全局模型
            # 客户利用替换后的全局模型进行本地训练
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)
            # 计算每个用户聚合时的权重
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            matrix = []
            for net_id, net in enumerate(nets_this_round.values()):
                net_vector = torch.nn.utils.parameters_to_vector(net.parameters())
                matrix.append(net_vector.tolist())
            matrix = np.array(matrix)
            # 计算median
            Median = np.ma.median(matrix, axis=0).data
            Median = Median.tolist()
            Median = torch.tensor(Median)

            torch.nn.utils.vector_to_parameters(Median, global_model.parameters())



            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Median Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Median Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Median Global Model Train loss: %f' % train_loss)
            print(" Median Global Model Train accuracy"+str(train_acc))
            print("Median Global Model Test accuracy"+str(test_acc))
    elif args.alg == 'trimmed_mean':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]#这一轮参与训练的所有客户

            global_w = global_model.state_dict()
            if args.server_momentum:#滑动平均
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}#nets_this_round本轮中参与训练的所有用户的模型
            for net in nets_this_round.values():
                net.load_state_dict(global_w)#将本地模型替换为全局模型

            #客户利用替换后的全局模型进行本地训练
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

            #计算每个用户聚合时的权重
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            matrix = []
            for net_id, net in enumerate(nets_this_round.values()):
                net_vector = torch.nn.utils.parameters_to_vector(net.parameters())
                matrix.append(net_vector.tolist())

            matrix = np.array(matrix)
            Trimmean = torch.tensor(matrix)
            sorted_result, indexs = torch.sort(Trimmean, dim=0)

            Trimmean = sorted_result[5:14]


            Trimmean = Trimmean.sum(axis=0)
            Trimmean = Trimmean.div(10)
            Trimmean = Trimmean.type(torch.float32)

            torch.nn.utils.vector_to_parameters(Trimmean, global_model.parameters())


            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            logger.info('>> trimmed_mean Global Model Train accuracy: %f' % train_acc)
            logger.info('>> trimmed_mean Global Model Test accuracy: %f' % test_acc)
            logger.info('>> trimmed_mean Global Model Train loss: %f' % train_loss)
            print(" trimmed Global Model Train accuracy" + str(train_acc))
            print("trimmed Global Model Test accuracy" + str(test_acc))
    elif args.alg == "krum":
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]#这一轮参与训练的所有客户
            global_w = global_model.state_dict()
            if args.server_momentum:#滑动平均
                old_w = copy.deepcopy(global_model.state_dict())
            nets_this_round = {k: nets[k] for k in party_list_this_round}#nets_this_round本轮中参与训练的所有用户的模型
            for net in nets_this_round.values():
                net.load_state_dict(global_w)#将本地模型替换为全局模型
            #客户利用替换后的全局模型进行本地训练
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)
            #计算每个用户聚合时的权重
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            matrix = []
            for net_id, net in enumerate(nets_this_round.values()):
                net_vector = torch.nn.utils.parameters_to_vector(net.parameters())
                matrix.append(net_vector.tolist())
            matrix = torch.tensor(matrix)
            # 计算matrix中每一个向量和其他向量的距离
            Distance = []
            for i in range(matrix.shape[0]):
                Distance_row = []
                for j in range(matrix.shape[0]):
                    if i == j:
                        Distance_row.append(0.0)
                    else:
                        temp = matrix[i] - matrix[j]
                        norm_value = torch.norm(temp, p=2, dim=0)
                        norm_value = torch.pow(norm_value, 2)
                        Distance_row.append(norm_value.item())
                Distance.append(Distance_row)
            Distance = torch.tensor(Distance)

            # 计算每一个向量i最近的n-f-2个向量的距离的加和
            Scores = []
            for i in range(Distance.shape[0]):
                values, indices = torch.topk(Distance[i], k=12, dim=0, largest=False)
                values = torch.sum(values, dim=0).item()
                Scores.append(values)
            Scores = torch.tensor(Scores)
            index_min = Scores.argmin().item()
            torch.nn.utils.vector_to_parameters(matrix[index_min], global_model.parameters())
            global_model.to(device)
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            logger.info(f"krum Global Model Train accuracy:  {train_acc}")
            logger.info(f"krum Global Model Test accuracy: {test_acc}")
            logger.info(f"krum Global Model Train loss:{train_loss}")