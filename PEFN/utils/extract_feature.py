import torch


def extract_feature(model, loader):
    with torch.no_grad():
        feats = []
        for (img, _) in loader:
            input_img = img.to('cuda')
            f1 = model(input_img)[-1].data.cpu()
            flip_img = img.index_select(3, torch.arange(img.size(3) - 1, -1, -1).long()).to('cuda')#得到逆向序列，就可以水平翻转
            f2 = model(flip_img)[-1].data.cpu()
            f = f1+ f2
            fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
            f = f.div(fnorm.expand_as(f))
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        return feats
#tensor.index_select(dim, index) → Tensor 可以得到tensor指定维度指定索引的值



