import torch, argparse, sys
from backdoor_attack import IPBA
sys.path.append("..")
import utils.general_utils as utils
import split_model


class ViewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="cifar10")
    parser.add_argument("--c", type=int, default=3000) # round
    parser.add_argument("--userdata", type=int, default=40000)
    parser.add_argument("--attackerdata", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=10.)
    parser.add_argument("--iid", type=str, default="SVHN")  # tiny_imagenet, cifar100, cifar10, SVHN
    args = parser.parse_args()

    utils.set_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    indices = torch.randperm(50000).tolist()

    client_loader, test_loader = utils.DataManger.get_dataloader(args.task, root="../data",
                                                                 batch_size=128,
                                                                 normalize=True,
                                                                 indices=[indices[:args.userdata],
                                                                          list(range(1000))])

    attacker_loader, _ = utils.DataManger.get_dataloader(args.iid, root="../data", batch_size=128,
                                                         normalize=True,
                                                         indices=[indices[
                                                                  args.userdata:args.userdata + args.attackerdata],
                                                                  list(range(10))])

    early_layer, intermediate_layer, late_layer = split_model.split("resnet", split_idx)
    early_layer, intermediate_layer, late_layer = early_layer.to(device), intermediate_layer.to(device), late_layer.to(device)
    shadow_layer = split_model.get_shadow_model("resnet", "single", split_idx).to(device) # multi

    ipba = IPBA([early_layer, intermediate_layer, late_layer], shadow_layer,
                client_loader, attacker_loader, test_loader, args.alpha, device)

    accuracy, misclassification_rate = ipba.train(args.round)
