import random
import collections
from torch.utils.data import sampler
   
class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id

        self._id2index = collections.defaultdict(list)#如此，value就是一个列表，可以直接append，把同id的车辆的索引放在一起
        for idx, _id in enumerate(data_source.pids):#{10:0,20:1,22:2}
            self._id2index[_id].append(idx)

    def __iter__(self):#依次返回每个样本的索引
        unique_ids = list(self._id2index.keys())
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image

    @staticmethod
    def _sample(population, k):
        #random.seed(42)
        if len(population) < k:
            population = population * k
        return random.sample(population, k)
