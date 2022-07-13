from functools import partial
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .tokenization_bert import BertTokenizer
from .vit import VisionTransformer, interpolate_pos_embed
from .xbert import BertConfig, BertModel


class ALBEF(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(config['bert_dir'])
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=config['cache_dir'])
        self.distill = config['distill']
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), checkpoint=config['checkpoint_vit'])

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(config['bert_dir'], config=bert_config, add_pooling_layer=False)
        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=bert_config, add_pooling_layer=False,
        #                                         cache_dir=config['cache_dir'])

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)

        if self.distill:
            # create momentum models
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.vision_proj_m = nn.Linear(vision_width, embed_dim)
            # self.text_encoder_m = BertModel.from_pretrained('bert-base-uncased', config=bert_config,
            #                                     cache_dir=config['cache_dir'], add_pooling_layer=False)
            self.text_encoder_m = BertModel.from_pretrained(config['bert_dir'], config=bert_config,
                                                            add_pooling_layer=False)
            self.text_proj_m = nn.Linear(text_width, embed_dim)

            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.vision_proj, self.vision_proj_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_proj, self.text_proj_m],
                                ]
            self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    @classmethod
    def from_cktp(cls, config, ckpt_path=None):
        model = cls(config)

        assert ckpt_path is not None or 'checkpoint_path' in config.keys()
        ckpt_path = ckpt_path if ckpt_path is not None else config['checkpoint_path']
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if config['distill']:
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

        # model.queue_size = 65472
        # model.image_queue = model.image_queue[:, :65472]
        # model.text_queue = model.text_queue[:, :65472]
        # model.idx_queue = model.idx_queue[:, :65472]

        return model

    @torch.no_grad()
    def matching(self, image, text):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   mode='multi_modal')

        score = self.itm_head(output.last_hidden_state[:, 0, :])
        score = torch.softmax(score, dim=-1)
        return score[:, 1]

    @torch.no_grad()
    def similarity_and_matching(self, image, text, pairwise=False) -> Dict:
        """
        Returns the similarity and matching score of image and text. The matching score is between 0 and 1, where 0 is
        a bad matching and 1 indicates a very good matching.
        @param image: Tensor of batched images (bs, c, h, w)
        @param text: Tokenized text, must contain input_ids and attention_mask
        @param pairwise: if false computes the matching score between one image and one text. if true all pairs were
        computed
        """
        image_embeds = self.visual_encoder(image)
        image_feat = self.vision_proj(image_embeds[:, 0, :])
        image_feat = F.normalize(image_feat, dim=-1)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, output_hidden_states=True, return_dict=True,
                                        mode='text')
        text_embeds = text_output.last_hidden_state
        text_hidden_states = torch.stack(text_output.hidden_states)
        text_feat = self.text_proj(text_embeds[:, 0, :])
        text_feat = F.normalize(text_feat, dim=-1)

        sim_i2t = image_feat @ text_feat.t() / self.temp
        sim_t2i = text_feat @ image_feat.t() / self.temp

        if pairwise:
            all_scores = []
            for i in range(image_embeds.shape[0]):
                encoder_output = image_embeds[i].repeat(text_embeds.shape[0], 1, 1)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(encoder_output.device)

                output = self.text_encoder(encoder_embeds=text_embeds,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=encoder_output,
                                           encoder_attention_mask=encoder_att,
                                           return_dict=True,
                                           output_hidden_states=True,
                                           mode='fusion')
                score = self.itm_head(output.last_hidden_state[:, 0, :])
                all_scores.append(torch.softmax(score, dim=-1)[:, 1])
            multi_hidden_states = torch.stack(output.hidden_states)
            score = torch.stack(all_scores)
        else:
            output = self.text_encoder(encoder_embeds=text_embeds,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       mode='fusion')

            score = self.itm_head(output.last_hidden_state[:, 0, :])
            score = torch.softmax(score, dim=-1)[:, 1]
        hidden_states = {'txt': text_hidden_states[1:], 'multi': multi_hidden_states[1:]}
        c_out = {'txt': text_feat, 'img': image_feat}
        return dict(score=score, i2t=sim_i2t, t2i=sim_t2i, hs=hidden_states, c_out=c_out)

    @torch.no_grad()
    def _similarity_and_matching(self, image, text_embeds, text_hidden_states, text_attention_mask, pairwise=False) -> Dict:
        """
        Returns the similarity and matching score of image and text. The matching score is between 0 and 1, where 0 is
        a bad matching and 1 indicates a very good matching.
        @param image: Tensor of batched images (bs, c, h, w)
        @param text: Tokenized text, must contain input_ids and attention_mask
        @param pairwise: if false computes the matching score between one image and one text. if true all pairs were
        computed
        """
        image_embeds = self.visual_encoder(image)
        image_feat = self.vision_proj(image_embeds[:, 0, :])
        image_feat = F.normalize(image_feat, dim=-1)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text_feat = self.text_proj(text_embeds[:, 0, :])
        text_feat = F.normalize(text_feat, dim=-1)

        sim_i2t = image_feat @ text_feat.t() / self.temp
        sim_t2i = text_feat @ image_feat.t() / self.temp

        if pairwise:
            all_scores = []
            for i in range(image_embeds.shape[0]):
                encoder_output = image_embeds[i].repeat(text_embeds.shape[0], 1, 1)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(encoder_output.device)

                output = self.text_encoder(encoder_embeds=text_embeds,
                                           attention_mask=text_attention_mask,
                                           encoder_hidden_states=encoder_output,
                                           encoder_attention_mask=encoder_att,
                                           return_dict=True,
                                           output_hidden_states=True,
                                           mode='fusion')
                score = self.itm_head(output.last_hidden_state[:, 0, :])
                all_scores.append(torch.softmax(score, dim=-1)[:, 1])
            multi_hidden_states = torch.stack(output.hidden_states)
            score = torch.stack(all_scores)
        else:
            output = self.text_encoder(encoder_embeds=text_embeds,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       mode='fusion')

            score = self.itm_head(output.last_hidden_state[:, 0, :])
            score = torch.softmax(score, dim=-1)[:, 1]
        hidden_states = torch.vstack((text_hidden_states, multi_hidden_states))
        return dict(score=score, i2t=sim_i2t, t2i=sim_t2i, hs=hidden_states)

    def forward(self, image, text, alpha, idx):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            if self.distill:
                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        if self.distill:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        else:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder(encoder_embeds=text_embeds,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       mode='fusion',
                                       )
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs] + 1e-4, dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs] + 1e-4, dim=1)

            mask = torch.eq(idx, idx.T)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

            # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(encoder_embeds=text_embeds_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True,
                                       mode='fusion',
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_ita, loss_itm

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)

        assert self.queue_size % batch_size == 0, f'{self.queue_size=}, but {batch_size=}'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
