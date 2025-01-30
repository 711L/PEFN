from torch.optim import Adam, SGD
from CV.PVReid.opt1 import opt1


def get_optimizer(net):
    if opt1.freeze:

        for p in net.parameters():
            p.requires_grad = True
        for q in net.backbone.parameters():
            q.requires_grad = False

        optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opt1.lr, weight_decay=5e-4,
                         amsgrad=True)

    else:

       # optimizer = SGD(net.parameters(), lr=opt.lr,momentum=0.9, weight_decay=5e-4)
        optimizer = Adam(net.parameters(), lr=opt1.lr, weight_decay=5e-4, amsgrad=True)

    return optimizer
