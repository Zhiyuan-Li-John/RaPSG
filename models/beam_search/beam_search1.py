import torch
import utils
from torch.nn import functional as F

class BeamSearch1(object):
    '''add segmentation feature'''
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
            visual_exp = visual.view(visual_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual

    def apply(self, clip_img, visual: utils.TensorOrSequence, pixels: utils.TensorOrSequence, out_size=1, return_probs=False, **kwargs):
        self.b_s = utils.get_batch_size(visual)
        self.device = utils.get_device(visual)
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        outputs = []
        contrastive_loss_all = 0
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                visual, outputs, contrastive_loss, clip_img = self.iter(clip_img, t, visual, pixels, outputs, return_probs, **kwargs)
                contrastive_loss_all += contrastive_loss

        # Sort result
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                                          self.max_len,
                                                                                          all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs, contrastive_loss_all

    def select(self, t, candidate_logprob, **kwargs):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        return selected_idx, selected_logprob

    def iter(self, clip_img, t: int, visual: utils.TensorOrSequence, pixels: utils.TensorOrSequence, outputs, return_probs, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size

        word_logprob = self.model.step(t, self.selected_words, visual, pixels, None, mode='feedback', **kwargs)
        [a, b, c] = word_logprob.shape
        semi_word = word_logprob.view(a, c)
        semi_word = F.normalize(semi_word)
        semi_visual = F.normalize(clip_img)  # semi_visual
        semi_word = torch.mean(semi_word, dim=1)
        semi_visual = torch.mean(semi_visual, dim=1)
        semi_word = semi_word.view(a, 1)
        semi_visual = semi_visual.view(a, 1)
        logits1 = torch.mm(semi_word, semi_visual.t())
        logits2 = torch.mm(semi_visual, semi_word.t())
        tau = 0.2  # temperature
        [N, N] = logits1.shape
        labels = torch.zeros(N, dtype=torch.int64).to(torch.device('cuda'))
        contrastive_loss = F.cross_entropy(logits1 / tau, labels) + F.cross_entropy(logits2 / tau, labels)
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)
        candidate_logprob = self.seq_logprob + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)

        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs)
        selected_beam = selected_idx // candidate_logprob.shape[-1]  # revise
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))
        visual = self._expand_visual(visual, cur_beam_size, selected_beam)
        clip_img = self._expand_visual(clip_img, cur_beam_size, selected_beam)

        self.seq_logprob = selected_logprob.unsqueeze(-1)
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
        outputs.append(selected_words.unsqueeze(-1))

        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        self.log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs)
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)

        return visual, outputs, contrastive_loss, clip_img

