import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
import dataset, vgg, feature

def main():
    parser = argparse.ArgumentParser(description='VGG16 Channel Feature Map Example')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--model_root', type=str, help='folder to save the model')
    parser.add_argument('--data_root', type=str, help='folder to save the dataset')
    parser.add_argument('--input_size', type=int, default=224, help='input size of image')
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size (default: 50)')
    parser.add_argument('--n_batch', type=int, default=2, help='number of batches to infer')
    args = parser.parse_args()
    
    print("Emptying GPU caches")
    torch.cuda.empty_cache()
    
    project_dir = Path(__file__).resolve().parent.parent
    args.model_root = project_dir / 'ckpt' / 'visualized'
    args.data_root = project_dir / 'data' / 'imagenet-val'
    
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    print("Building and initializing vgg16 parameters")
    m = vgg.vgg16(pretrained=True, model_root=args.model_root)
    if torch.cuda.is_available():
        model_raw = m.cuda()
        
    project_dir = Path(__file__).resolve().parent.parent
    ds_fetcher = dataset.get #rename func

    # eval model
    val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    
    acc1, acc5 = feature.eval_model(model_raw, val_ds, args.n_batch)

    # print sf
    print(model_raw)
    res_str = "vgg16, acc1={:.4f}, acc5={:.4f}".format(acc1, acc5)
    print(res_str)
    with open('acc1_acc5.txt', 'a') as f:
        f.write(res_str + '\n')
    
    print("Emptying GPU caches")
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
