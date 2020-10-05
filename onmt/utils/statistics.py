""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
import torch

from onmt.utils.logging import logger


class Statistics(object):
    """
    Accumulator for total_loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time

    Add:
    * total_loss
    * hard_loss
    * soft_loss
    * internal_loss
    """

    def __init__(self, total_loss=0, hard_loss=0, soft_loss=0, n_words=0, n_correct=0, counter=0):
        self.total_loss = total_loss
        self.hard_loss = hard_loss
        self.soft_loss = soft_loss
        self.enc_loss = 0.0
        self.dec_loss = 0.0
        self.attn_loss = 0.0
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()
        self.counter = counter

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from onmt.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.total_loss += stat.total_loss
        self.hard_loss += stat.hard_loss
        self.soft_loss += stat.soft_loss
        self.enc_loss += stat.enc_loss
        self.dec_loss += stat.dec_loss
        self.attn_loss += stat.attn_loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.counter += stat.counter

        if update_n_src_words:
            self.n_src_words += stat.n_src_words
    
    def update_enc_loss (self, enc_loss, beta):
        if isinstance(enc_loss, torch.Tensor):
            self.enc_loss = enc_loss.clone().item()  # update enc loss here
            self.total_loss += self.enc_loss * beta       # update the total_loss
    
    def update_dec_loss (self, dec_loss, beta):
        if isinstance(dec_loss, torch.Tensor):
            self.dec_loss = dec_loss.clone().item()  # update dec loss here
            self.total_loss += self.dec_loss * beta       # update the total_loss
    
    
    def update_attn_loss (self, attn_loss, theta):
        if isinstance(attn_loss, torch.Tensor):
            self.attn_loss = attn_loss.clone().item()  # update attn_loss here
            self.total_loss += self.attn_loss * theta       # update the total_loss


    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.total_loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.total_loss / self.n_words, 100))
    
    def hard_loss_perword(self):
        """ compute hard loss per word """
        return self.hard_loss / self.n_words
    
    def soft_loss_perword(self):
        """  compute soft loss per word """
        return self.soft_loss / self.n_words
    
    def enc_loss_perword(self):
        """ compute internal loss per word """
        return self.enc_loss / self.n_words
    
    def dec_loss_perword(self):
        """ compute internal loss per word """
        return self.dec_loss / self.n_words
    
    def attn_loss_perword(self):
        """ computer attn loss per word """
        return self.attn_loss / self.n_words


    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("Step %s; acc: %5.2f; ppl: %5.2f; xent: %4.2f; " +
             "hard_loss:%5.2f; soft_loss:%5.2f; enc_loss:%5.2f; dec_loss:%5.2f; attn_loss:%5.2f; lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step_fmt,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               self.hard_loss_perword(),
               self.soft_loss_perword(),
               self.enc_loss_perword(),
               self.dec_loss_perword(),
               self.attn_loss_perword(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/hard_loss", self.hard_loss_perword(),step)
        writer.add_scalar(prefix + "/soft_loss", self.soft_loss_perword(),step)
        writer.add_scalar(prefix + "/enc_loss", self.enc_loss_perword(),step)
        writer.add_scalar(prefix + "/dec_loss", self.dec_loss_perword(),step)
        writer.add_scalar(prefix + "/attn_loss", self.attn_loss_perword(),step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
