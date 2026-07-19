import argparse
import torch
import mdg
import numpy as np

log = mdg.utils.get_logger()



def main(args):
    mdg.utils.set_seed(args.seed)

    log.debug("Loading data from '%s'." % args.data)
    data = mdg.utils.load_pkl(args.data)
    log.info("Loaded data.")

    testset = mdg.Dataset(data["test"], args.batch_size)
    # testset = mdg.Dataset(data["dev"], args.batch_size)


    # model_file = "./save/best_model.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/full_TF_MA_RGCN_GT_GSAGE_0634.pt"

    model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_BLSTM_MA_RGCN_GT_GSAGE_0495.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_BGRU_MA_RGCN_GT_GSAGE_0421.pt"

    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_TF_cat_RGCN_GT_GSAGE_0587.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_TF_MA_GCN_GT_GSAGE_0603.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_TF_MA_RGCN_GT_0615.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_TF_MA_RGCN_GSAGE_0621.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_TF_MA_GT_GSAGE_0614.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_TF_MA_RGCN_0547.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_TF_MA_GSAGE_0560.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_TF_MA_GT_0598.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_TF_MA_FC_0523.pt"

    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_CCC_MSE_0615.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_CCC_VAR_0608.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_MSE_VAR_0.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_CCC_0561.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_MSE_0.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_VAR_0.pt"

    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_Nhead_6_0605.pt"
    # model_file = "E:/PCProjects/MMDD/DepMMGNN_code_clean/save/ablation_Nw_50_0611.pt"
    
    print(f"Loading model from '{model_file}'.")
    model = mdg.MDG(args).to(args.device)

    pred = mdg.Prediction(testset, model, args)
    ckpt = torch.load(model_file)
    pred.load_ckpt(ckpt)

    gt, preds = pred.pred()
    np.save('./res/gt.npy', gt)
    np.save('./res/preds.npy', preds)

    print(gt)
    print(preds)

    gt = torch.tensor(gt)
    preds = torch.tensor(preds)
    out_mean = torch.mean(gt)
    target_mean = torch.mean(preds)

    covariance = torch.mean( (gt - out_mean) * (preds - target_mean) )
    target_var = torch.mean( (preds - target_mean)**2)
    out_var = torch.mean( (gt - out_mean)**2 )

    ccc = 2.0 * covariance/(target_var + out_var + (target_mean-out_mean)**2 + 1e-10)
    rmse = torch.sqrt(torch.mean((gt - preds)**2))

    print(f"CCC: {ccc:.4f}, RMSE: {rmse:.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="prediction.py")
    parser.add_argument("--data", type=str, required=True, help="Path to data")

    parser.add_argument("--device", type=str, default="cpu", help="Computing device.")

    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs.")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")


    # Model parameters
    parser.add_argument("--drop_rate", type=float, default=0.5, help="Dropout rate.")

    parser.add_argument("--wp", type=int, default=10, help="Past context window size. Set wp to -1 to use all the past context.")
    parser.add_argument("--wf", type=int, default=10, help="Future context window size. Set wp to -1 to use all the future context.")

    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers.")

    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden size of two layer GCN.")

    parser.add_argument("--rnn", type=str, default="lstm", choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")
    # parser.add_argument("--rnn", type=str, default="gru", choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")
    # parser.add_argument("--rnn", type=str, default="transformer", choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")

    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    
    parser.add_argument("--scale_para", type=float, default=-0.5, help="attention scale para.")
    
    args = parser.parse_args()
    log.debug(args)

    main(args)