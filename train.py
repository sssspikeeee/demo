import argparse

import torch

import mdg

log = mdg.utils.get_logger()


def main(args):
    mdg.utils.set_seed(args.seed)

    # 1. 加载数据
    log.debug("Loading data from '%s'." % args.data)
    data = mdg.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = mdg.Dataset(data["train"], args.batch_size)
    devset = mdg.Dataset(data["dev"], args.batch_size)
    testset = mdg.Dataset(data["test"], args.batch_size)

    # 2. 初始化模型、优化器
    log.debug("Building model...")
    model_file = "./save/best_model.pt"
    model = mdg.MDG(args).to(args.device)
    opt = mdg.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)

    # 3. 创建Coach训练管理器 
    coach = mdg.Coach(trainset, devset, testset, model, opt, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)

    # 4. 执行训练循环
    log.info("Start training...")
    parameters = coach.count_parameters()
    log.info("Total parameters: %d" % parameters)
    ret = coach.train()

    # 5.保存最佳模型
    checkpoint = {
        "best_dev_ccc": ret[0],
        "best_test_ccc": ret[1],
        "best_dev_rmse": ret[2],
        "best_test_rmse": ret[3],
        "best_epoch": ret[4],
        "best_state": ret[5],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")

    # Training parameters
    parser.add_argument("--from_begin", action="store_true", help="Training from begin.")

    parser.add_argument("--device", type=str, default="cpu", help="Computing device.")

    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs.")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")

    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "rmsprop", "adam"], help="Name of optimizer.")

    # parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")

    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay.")

    parser.add_argument("--max_grad_value", default=-1, type=float, help="If the norm of the gradient vector exceeds this, normalize it to have the norm equal to max_grad_norm.")

    parser.add_argument("--drop_rate", type=float, default=0.5, help="Dropout rate.")

    parser.add_argument("--scale_para", type=float, default=-0.5, help="attention scale para.")

    # Model parameters

    # parser.add_argument("--wp", type=int, default=5, help="Past context window size. Set wp to -1 to use all the past context.")
    # parser.add_argument("--wf", type=int, default=5, help="Future context window size. Set wf to -1 to use all the future context.")

    parser.add_argument("--wp", type=int, default=10, help="Past context window size. Set wp to -1 to use all the past context.")
    parser.add_argument("--wf", type=int, default=10, help="Future context window size. Set wf to -1 to use all the future context.")

    # parser.add_argument("--wp", type=int, default=15, help="Past context window size. Set wp to -1 to use all the past context.")
    # parser.add_argument("--wf", type=int, default=15, help="Future context window size. Set wf to -1 to use all the future context.")

    # parser.add_argument("--wp", type=int, default=20, help="Past context window size. Set wp to -1 to use all the past context.")
    # parser.add_argument("--wf", type=int, default=20, help="Future context window size. Set wf to -1 to use all the future context.")
    
    # parser.add_argument("--wp", type=int, default=25, help="Past context window size. Set wp to -1 to use all the past context.")
    # parser.add_argument("--wf", type=int, default=25, help="Future context window size. Set wf to -1 to use all the future context.")

    # parser.add_argument("--wp", type=int, default=30, help="Past context window size. Set wp to -1 to use all the past context.")
    # parser.add_argument("--wf", type=int, default=30, help="Future context window size. Set wf to -1 to use all the future context.")

    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers.")

    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden size of two layer GCN.")

    # parser.add_argument("--rnn", type=str, default="lstm", choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")
    # parser.add_argument("--rnn", type=str, default="gru", choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")
    parser.add_argument("--rnn", type=str, default="transformer", choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    log.debug(args)

    main(args)

