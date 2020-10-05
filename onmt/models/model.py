""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def _detach(obj):
        """
            If obj is pytorch Tensor then detach is applied,
            otherwise, for dict and list recusrive call is made
            any other type error is raised
        """

        if isinstance(obj, torch.Tensor):
            return obj.detach()
        elif isinstance(obj, dict):
            for key, val in obj.items():
                obj[key] = NMTModel._detach(val)
        elif isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = NMTModel._detach(obj[i])
        else:
            raise ValueError(f'Cannot detach object of type {type(obj)}')
            

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, queue=None, input_queue=None, enc_distillation_mode=None, dec_distillation_mode=None, enc_step=1, dec_step=1, student_dim=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.
            queue (torch.multiprocessing.Queue): queue in which Tensors to be returned will be added
            input_queue (torch.multiprocessing.Queue): queue containing torch Tensors for input
            enc_distillation_mode: Do which distillation on encoder
            dec_distillation_mode: Do which distillation on decoder
            enc_step (int): For PKD the step size for teacher encoder for intermediate outputs
            dec_step (int): For PKD the step size for teacher decoder for intermediate outputs
        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """

        if input_queue:
            src = input_queue.get()
            tgt = input_queue.get()
            lengths = input_queue.get()
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths, enc_intermediate, enc_attns = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)

        # dec_state: embedding
        dec_state, dec_out, attns, dec_intermediate, unnormalized_attns, dec_attns = self.decoder(dec_in, memory_bank,
                                                        memory_lengths=lengths,
                                                        with_align=with_align)
        # list [bs*len*hid_dim] * layers
        enc_intermediate = NMTModel._stack_internals(enc_intermediate, enc_step, enc_distillation_mode)         
        dec_intermediate = NMTModel._stack_internals(dec_intermediate, dec_step, dec_distillation_mode)
        # for attention distillation only support PKD now
        unnormalized_attns = NMTModel._stack_internals(unnormalized_attns, dec_step,'skip')
        enc_attns = NMTModel._stack_internals(enc_attns, enc_step, 'skip')
        dec_attns = NMTModel._stack_internals(dec_attns, dec_step, 'skip')        
 
        if enc_distillation_mode or dec_distillation_mode:
            if queue:
                queue.put(NMTModel._detach(dec_out))
                queue.put(NMTModel._detach(attns))
                queue.put(NMTModel._detach(enc_intermediate))
                queue.put(NMTModel._detach(dec_intermediate))
                queue.put(NMTModel._detach(unnormalized_attns))
                queue.put(NMTModel._detach(enc_attns))
                queue.put(NMTModel._detach(dec_attns))
                queue.close()
            else:
                return dec_out, attns, enc_intermediate, dec_intermediate, unnormalized_attns, enc_attns, dec_attns
        else:
            return dec_out, attns, enc_state, dec_state, enc_intermediate, dec_intermediate, unnormalized_attns, enc_attns, dec_attns

    @staticmethod
    def _stack_internals(tensor_list, step=1, distill_mode=None):
        # for student step is always set to 1
        if distill_mode == 'skip':
            new_tensor_list = tensor_list[step-1::step] 
        elif distill_mode == 'regular_comb':
            l = len(tensor_list)
            new_tensor_list = []
            for i in range(0,l,step):
                temp = torch.cat(tensor_list[i:i+step],dim=-1)   # concat layer
                new_tensor_list.append(temp)
        # hard code (only 6 layer -> 2 layer)
        elif distill_mode == 'overlap': 
            layer_1 = torch.cat(tensor_list[0:4],dim=-1) #layer: [1,2,3,4]
            layer_2 = torch.cat(tensor_list[2::],dim=-1) #layer: [3,4,5,6]
            new_tensor_list = [layer_1,layer_2]
        
        elif distill_mode == 'skip_middle':
            layer_1 = torch.cat(tensor_list[0:2],dim=-1)  #layer:[1,2]
            layer_2 = torch.cat(tensor_list[4::],dim=-1)  #layer:[5,6]
            new_tensor_list = [layer_1,layer_2]
        
        elif distill_mode == 'cross_comb':
            layer_1 = torch.cat(tensor_list[0:3:2],dim=-1)  #layer: [1,3]
            layer_2 = torch.cat(tensor_list[3::2],dim=-1)   #layer: [4,6]
            new_tensor_list = [layer_1,layer_2]
        
        elif distill_mode == 'skip_comb':
            layer_1 = torch.cat(tensor_list[0:5:2],dim=-1)  #layer: [1,3,5]
            layer_2 = torch.cat(tensor_list[1:6:2],dim=-1)  #layer: [2,4,6]
            new_tensor_list = [layer_1,layer_2]
                            
        else:
            new_tensor_list = tensor_list[step-1::step] 
                    
        return new_tensor_list
    
    ## TODO: stack attention

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
