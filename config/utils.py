import os
from copy import deepcopy


def get_outdir(cfg, root_dir='./out'):
    cfg = deepcopy(cfg)

    def _append_extract_layers(extract_layers):
        append_name = ''
        for extract_layer in extract_layers:
            append_name += (extract_layer + '-')
        # remove last `-`
        return append_name[:-1]

    # add information about stages when multi-stage network
    if cfg['meta_architecture'] == 'MultiStageNet':
        cfg['meta_architecture'] = \
            os.path.join(cfg['meta_architecture'],
                         'Stage{}'.format(len(cfg['estimator'])))
    if cfg['extractor_freeze']:
        cfg['extractor'] += '_freeze'
    if 'val_img_dir' in cfg.keys() and 'val_keypoints_dir' in cfg.keys():
        cfg['extractor'] += '_with_val'
    if 'outpath_additional' in cfg.keys():
        cfg['extractor'] += cfg['outpath_additional']

    outpath = os.path.join(
        root_dir, cfg['meta_architecture'], cfg['extractor'], cfg['loss'],
        _append_extract_layers(cfg['extract_layers']))
    return outpath
