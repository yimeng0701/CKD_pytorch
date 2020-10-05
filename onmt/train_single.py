#!/usr/bin/env python
"""Training on a single process."""
import os

import torch

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser



def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    others = 0
    trainable_paras = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name or 'generator' in name:
            dec += param.nelement()
        else:
            others += param.nelement() 
        if param.requires_grad:
            trainable_paras += param.nelement()
    return enc+dec+others, enc, dec, others, trainable_paras


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)

def layer_projection(model_opt,distillation_part):
    if distillation_part == 'encoder':
        num_layers = model_opt.enc_layers
        distillation_mode = model_opt.enc_distillation_mode
    else:
        num_layers = model_opt.dec_layers
        distillation_mode = model_opt.dec_distillation_mode 
    linear = None

    # Build linear layer for dim matching between teacher and student
    if distillation_mode == 'regular_comb':
        linear = torch.nn.ModuleList([torch.nn.Linear(model_opt.rnn_size, model_opt.rnn_size*3)
                                                    for _ in range(num_layers)])
    elif distillation_mode == 'overlap': 
        linear = torch.nn.ModuleList([torch.nn.Linear(model_opt.rnn_size, model_opt.rnn_size*4)
                                                    for _ in range(num_layers)])
    
    elif distillation_mode == 'skip_middle':
        linear = torch.nn.ModuleList([torch.nn.Linear(model_opt.rnn_size, model_opt.rnn_size*2)
                                                    for _ in range(num_layers)])
    
    elif distillation_mode == 'cross_comb':
        linear = torch.nn.ModuleList([torch.nn.Linear(model_opt.rnn_size, model_opt.rnn_size*2)
                                                    for _ in range(num_layers)])
    
    elif model_opt.enc_distillation_mode == 'skip_comb':
        linear = torch.nn.ModuleList([torch.nn.Linear(model_opt.rnn_size, model_opt.rnn_size*3)
                                                    for _ in range(num_layers)])
    return linear


def main(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(log_file=opt.log_file)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    teacher_model = None
    teacher_opt = None
    # Load teacher model if applying KD
    if opt.logit_distillation or opt.enc_distillation_mode or opt.dec_distillation_mode:
        logger.info('Loading teacher model from %s' % opt.teacher)
        teacher_checkpoint = torch.load(opt.teacher,
                                        map_location=lambda storage, loc: storage)
        teacher_model = build_model(teacher_checkpoint['opt'], opt, fields, teacher_checkpoint)
        teacher_opt = ArgumentParser.ckpt_model_opts(teacher_checkpoint['opt'])

        # Build linear layer for dim matching between teacher and student
        enc_linear = layer_projection(model_opt, distillation_part='encoder')
        dec_linear = layer_projection(model_opt, distillation_part='decoder')
                
        model.enc_linear = enc_linear
        model.dec_linear = dec_linear
        
        if model.enc_linear:
            model.enc_linear.to(device_id)
        
        if model.dec_linear:
            model.dec_linear.to(device_id)
        
        teacher_model.eval()
        teacher_model.generator.eval()
        if opt.kd_async:
          teacher_model.share_memory()
    
    
    
    n_params, enc, dec, others, trainable = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('layer_projection: %d' % others)
    logger.info('* number of parameters: %d' % n_params)
    logger.info('* number of trainable parameters: %d' % trainable)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, teacher_opt, teacher_model, fields, optim, model_saver=model_saver)

    if batch_queue is None:
        if len(opt.data_ids) > 1:
            train_shards = []
            for train_id in opt.data_ids:
                shard_base = "train_" + train_id
                train_shards.append(shard_base)
            train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
        else:
            if opt.data_ids[0] is not None:
                shard_base = "train_" + opt.data_ids[0]
            else:
                shard_base = "train"
            train_iter = build_dataset_iter(shard_base, fields, opt)

    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                yield batch

        train_iter = _train_iter()

    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
