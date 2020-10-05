#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.utils.tokenization import get_detok_func
import glob
import os
import torch
import subprocess
import yaml

def calc_bleu(hyp, ref, bleu_script, detok=None, params={}):
    if detok:
        detok_fn = get_detok_func(detok)
        detok_fn(hyp, hyp + '.detok', **params)

    cmd = ['bash', bleu_script, hyp + '.detok', ref]
    try:
        cmd_out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT)
        bleu_score = float(cmd_out.decode("utf-8"))
    except Exception as error:
        bleu_score = float(0.0)
    return bleu_score


# python tools/sacrebleu.py -i ${DATA_PATH}/output_eval $TEST_REF


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)
    val_file_path = os.path.join(opt.train_dir, opt.val_result_file)

    with open(val_file_path, 'r') as f:
        val_results = yaml.load(f)

    all_ckpts = []
    for key, result in val_results.items():
        loss = result['Validation loss']
        ckpt = opt.model_prefix + '_' + key.split()[-1] + '.pt'
        all_ckpts.append((ckpt, loss))

    all_ckpts.sort(key=lambda x: x[1])

    assert len(all_ckpts)>=opt.best_n_ckpts
    all_ckpts = all_ckpts[:opt.best_n_ckpts]

    print('Evaluating the following checkpoints')
    print(all_ckpts)
    local_ckpts = []

    for ckpt in all_ckpts:
        local_ckpt = os.path.join(opt.train_dir, ckpt[0])
        local_ckpts.append(local_ckpt)


    opt.models = local_ckpts

    train_config = {}
    checkpoint = torch.load(local_ckpts[0], map_location=lambda storage, loc: storage)
    train_config = checkpoint.get('train_config')
    config = checkpoint.get('opt')
    print(config)

    eval_dir_local = os.path.join(opt.train_dir, 'eval')
    data_dir_local = opt.data_dir
    if os.path.exists(eval_dir_local) is False:
        os.mkdir(eval_dir_local)

    vocab_model = os.path.join(data_dir_local, opt.vocab_model)
    source_files = opt.src.split(' ')
    target_files = opt.tgt.split(' ')
    results = {
        "datasets": [],
        "configs": {
            "detok_model": opt.vocab_model,
            "detok": opt.detok_type,
            "models": opt.models,
            "max_dec_length": opt.max_length,
            "train_config": train_config,
        }
    }

    for src, tgt in zip(source_files, target_files):
        opt.src = os.path.join(data_dir_local, src)
        opt.tgt = os.path.join(data_dir_local, tgt)
        opt.output = os.path.join(eval_dir_local, src + '.output')

        translator = build_translator(opt, report_score=True)
        src_shards = split_corpus(opt.src, opt.shard_size)
        tgt_shards = split_corpus(opt.tgt, opt.shard_size)
        shard_pairs = zip(src_shards, tgt_shards)

        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug
            )
        
        bleu_score = calc_bleu(opt.output, opt.tgt, opt.bleu_script, opt.detok_type, {'spm_model': vocab_model})
        results["datasets"].append({'src': src, 'tgt': tgt, 'bleu_score': bleu_score})

    results_file = os.path.join(eval_dir_local, 'results.yml')
    with open(results_file, 'w') as f:
        f.write(yaml.dump(results))
    print(results)


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()
    print('Following arguments are unrecognized')
    print(unknown)

    translate(opt)


if __name__ == "__main__":
    main()
