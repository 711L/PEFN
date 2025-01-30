import argparse

parser = argparse.ArgumentParser(description='vehicle reid settings')

parser.add_argument('--data_path',
                    default="/home/xxxxxx/datas/VeRi",
                    help='path of VeRi/vehilceID/VERI-WILD ')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate'],
                    help='train or evaluate ')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--device', default='1', type=str, help='gpus')
parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')
parser.add_argument('--save_path',default='./answer1.txt', help='Result Storage Path')
parser.add_argument('--weight',default='params/PEFN/.pt',help='load weights ')
parser.add_argument('--epoch',
                    default=700,
                    help='number of epoch to train')
parser.add_argument('--freeze', action='store_true',help="evaluation only")
parser.add_argument('--lr',default=2.6e-4,help='initial learning_rate')
parser.add_argument('--lr_scheduler',
                    default=[240,400,540,600],
                    help='MultiStepLR,decay the learning rate')
parser.add_argument('--resume',action='store_true',help="load weights")
parser.add_argument("--batchid",default=7,help='the batch for id')

parser.add_argument("--batchimage",default=9,help='the batch of per id')
parser.add_argument("--batchsize",default=63,help='the batch size for test')
parser.add_argument("--batchtest",default=24,help='the batch size for test')
parser.add_argument('--name', default='PEFN', type=str, help='gpus')
parser.add_argument('--dtype', default='vehicle', type=str, help='vehicle or other objects')
parser.add_argument('--data_name', default='veri', type=str, help='vehicleid, veri, veri-wild')
opt = parser.parse_args()
