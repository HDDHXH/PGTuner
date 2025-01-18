import  argparse

args = argparse.ArgumentParser()
args.add_argument('--vae-epochs', type=int, default=3000)#1000
args.add_argument('--dml-n-epochs', type=int, default=300)#300
args.add_argument('--predict-n-epochs', type=int, default=800)#800
args.add_argument('--dipredict-n-epochs', type=int, default=2000)#3000
args.add_argument('--dipredict-n-epochs-nsg', type=int, default=2000)#1000
args.add_argument('--predict-loss-n-epochs', type=int, default=3000)#1000
args.add_argument('--loss-n-epochs', type=int, default=3000)#1000
args.add_argument('--dml-batch-size', type=int, default=128)#128
args.add_argument('--predict-batch-size', type=int, default=2048)#1024
args.add_argument('--dipredict-batch-size', type=int, default=4096)#1024 512, 1024, 2048, 4096效果都差不多，不过还是1024效果更全面地好一点, 4096训练loss低很多，但预测效果反而差一些 有一个疑问是为什么训练集上loss很小，在测试及上反而误差增大了，还是得交叉验证
args.add_argument('--vae-batch-size', type=int, default=4096)
args.add_argument('--model', type=str, default='dim')#5种，分别是dml+knn（dml）、dml+mlp（dmm）、direct-mlp（dim）、mt-direct-mlp（mdim）、it-direct-mlp（idim）
args.add_argument('--dml-layer-sizes', type=str, default='[9, 128, 256, 256, 32]')#[9, 128, 128, 16], [9, 128, 128, 32], [9, 128, 128, 64]
args.add_argument('--predict-layer-sizes', type=str, default='[32, 256, 256, 64, 3]')
args.add_argument('--dipredict-layer-sizes', type=str, default='[14, 128, 256, 64, 3]') #[14, 128, 256, 64, 3]
args.add_argument('--dipredict-layer-sizes-nsg', type=str, default='[17, 128, 256, 64, 2]') #[14, 128, 256, 64, 3]
args.add_argument('--dipredict-conv-layer-sizes', type=str, default='[18, 128, 256, 64, 3]') #[8, 128, 256, 64, 3]
args.add_argument('--shared-layer-sizes', type=str, default='[9, 128, 256, 256, 64]')
args.add_argument('--private-layer-sizes', type=str, default='[64, 1]')
args.add_argument('--individual-layer-sizes', type=str, default='[9, 128, 256, 256, 64, 1]')
args.add_argument('--vae-layer-sizes', type=str, default='[14, 128, 256, 64, 14]') #[14, 128, 256, 64, 14]
args.add_argument('--inner-dim', type=int, default=32)#32
args.add_argument('--loss-lamb', type=float, default= 1.0)#1.0
args.add_argument('--margin', type=float, default= 1.0)#1.0
args.add_argument('--loss-margin', type=float, default= 0.0001)#0.01
args.add_argument('--loss', type=str, default='tl') #对比损失cl、三元组损失tl，不同的损失函数要构建不同的模型输入数据
args.add_argument('--miner', type=bool, default=True) #是否挖掘样本，如果为True，则使用半硬挖掘策略挖掘更有效的样本或三元组，这样可以减少输入样本对或者三元组的数量，降低显存占用，并且更有效
args.add_argument('--dml-lr', type=float, default=0.001)#0.001
args.add_argument('--predict-lr', type=float, default=0.001)#0.001
args.add_argument('--dipredict-lr', type=float, default=0.001)#0.001
args.add_argument('--predict-loss-lr', type=float, default=0.001)#0.001
args.add_argument('--vae-lr', type=float, default=0.001)#0.001
args.add_argument('--weight_decay', type=float, default=5e-4)#5e-4
args.add_argument('--threshold-t', type=float, default=0.9)#0.9
args.add_argument('--threshold-r', type=float, default=10)#1 暂定1，这个要看具体的数据，也不一定用
args.add_argument('--k', type=int, default=5)#检索k近邻
args.add_argument('--dml-valid-epoch', type=int, default=200)#200
args.add_argument('--predict-valid-epoch', type=int, default=700)#700
args.add_argument('--dipredict-valid-epoch', type=int, default=1900)#2900
args.add_argument('--dipredict-valid-epoch-nsg', type=int, default=1900)#900
args.add_argument('--vae-valid-epoch', type=int, default=1900)#900
args.add_argument('--predict-loss-valid-epoch', type=int, default=2900)#2900
args.add_argument('--max-count', type=int, default=5)#5
args.add_argument('--seed', type=int, default=42)#42

args, unknown = args.parse_known_args()
print(args)
