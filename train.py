import os
import argparse
import yaml
import chainer
from chainer import training
from chainer.training import extensions
from model import build_model, KeypointsDetector
from data.transforms import build_transforms
from data.keypoints_dataset import KeypointsDataset
from loss import build_loss
from config.utils import get_outdir
import matplotlib
matplotlib.use('Agg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to the config file of pose estimation.')
    parser.add_argument('device', type=int,
                        help='Device ID.')
    parser.add_argument('--progress', action='store_true',
                        help='Action is `store_true`.')
    args = parser.parse_args()
    return args


def freeze_extractor(model):
    for l in model.predictor.extractor.children():
        l.disable_update()


def train():
    args = parse_args()
    with open(args.config, 'r') as fp:
        cfg = yaml.load(fp)

    if 'val_img_dir' in cfg.keys() and 'val_keypoints_dir' in cfg.keys():
        flag_val = True
    else:
        flag_val = False
    if flag_val and 'early_stop' in cfg.keys() and cfg['early_stop']:
        flag_early_stop = True
    else:
        flag_early_stop = False

    # `outdir` is path to the directory saved log and model
    outdir = get_outdir(cfg)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # setup model
    detector = build_model(cfg)
    loss = build_loss(cfg, args.device)
    model = KeypointsDetector(detector, loss)

    if args.device >= 0:
        chainer.cuda.get_device_from_id(args.device).use()  # Make a specified GPU current
        model.to(args.device)

    # setup dataset
    transforms = build_transforms((224, 224), (56, 56), cfg['extractor'])
    train_dataset = KeypointsDataset(
        os.path.expanduser(cfg['train_img_dir']),
        os.path.expanduser(cfg['train_keypoints_dir']), transforms)

    optimizer = chainer.optimizers.SGD(lr=cfg['lr'])
    optimizer.setup(model)
    if cfg['extractor_freeze']:
        print('Freeze Extractor params: True')
        freeze_extractor(model)
    else:
        print('Freeze Extractor params: False')

    train_iter = chainer.iterators.SerialIterator(train_dataset, cfg['bs'])

    updater = training.updaters.StandardUpdater(
       train_iter, optimizer, device=args.device)

    if flag_val and flag_early_stop:
        print("Early Stopping: True")
        early_stop = training.triggers.EarlyStoppingTrigger(
            monitor='validation/main/loss', max_trigger=(cfg['epoch'], 'epoch'),
            patients=10)
        trainer = training.Trainer(updater, stop_trigger=early_stop, out=outdir)
    else:
        print("Early Stopping: False")
        trainer = training.Trainer(updater, (cfg['epoch'], 'epoch'), out=outdir)

    if flag_val:
        print('Validation: True')
        val_dataset = KeypointsDataset(
            os.path.expanduser(cfg['val_img_dir']),
            os.path.expanduser(cfg['val_keypoints_dir']), transforms)
        val_iter = chainer.iterators.SerialIterator(
            val_dataset, cfg['bs'], repeat=False, shuffle=False)
        trainer.extend(training.extensions.Evaluator(
            val_iter, model, device=args.device))
    else:
        print('Validation: False')

    # trainer extensions
    trainer.extend(extensions.LogReport())
    if flag_val:
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            'epoch', file_name='loss.png', marker=None))
    else:
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            'epoch', file_name='loss.png', marker=None))
    trainer.extend(extensions.snapshot(),
                   trigger=(cfg['step_save_model'], 'epoch'))
    if args.progress:
        trainer.extend(extensions.ProgressBar(update_interval=20))

    print('============= Start Training =============')
    trainer.run()
    print('============= End Training =============')


if __name__ == '__main__':
    train()
