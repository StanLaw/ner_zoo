'''
在这个文件定义需要的矩阵操作：
    1. 继承 EmbeddingOperation 这类, 命名为XXXX,实现init和_embed_op两个函数即可
            class XXXX(EmbeddingOperation):
            def __init__(self, config, model_args):
                添加需要的参数
            def _embed_op(self, outputs, attention_mask=None):
                bert输出向量如何操作
    2. 令BertDense类继承命名为XXXX的类
现网实现的例子：PoolingOperation。
'''
import logging
from typing import Union
import torch
import abc
from torch import nn, Tensor
import torch.nn.functional as F
#from transformers.activations import MishActivation
from transformers.activations import ACT2FN
from transformers import (
    AutoConfig, BertPreTrainedModel, BertModel,
    RobertaModel, RobertaPreTrainedModel
)
logger = logging.getLogger(__name__)


class EmbeddingOperation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _embed_op(self, outputs, attention_mask=None):
        pass
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

class LinearOperation(EmbeddingOperation):
    def __init__(self, config, model_args):
        self.model_args = model_args
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            # 默认使用192维输出
            self.output_embedding_size = 192
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        #self.norm = nn.LayerNorm(self.output_embedding_size)
    def _embed_op(self, outputs, attention_mask=None):
        text_embeds = outputs.last_hidden_state[:, 0]
        #norm is important in IP
        #text_embeds = self.norm(self.embeddingHead(text_embeds))
        text_embeds = self.embeddingHead(text_embeds)
        return text_embeds

class PoolingOperation(EmbeddingOperation):
    def __init__(self, config, model_args, output_embedding_size=192):
        self.model_args = model_args
        self.output_embedding_size = output_embedding_size
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.activation = ACT2FN['mish']
        #self.norm = nn.LayerNorm(self.output_embedding_size)
    def _embed_op(self, outputs, attention_mask=None):
        assert attention_mask is not None
        token_embeddings = self.activation(outputs.last_hidden_state)
        token_embeddings = self.embeddingHead(token_embeddings)
        return self.mean_pooling(token_embeddings, attention_mask)
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_over_time = torch.max(token_embeddings, 1)[0]
        return max_over_time
class GeneralOperation(EmbeddingOperation):
    '''
    这个类自动加载现网mean-pooling的bert,以及带简单线性输出层的bert。
    大部分情况用这个即可
    '''
    def __init__(self, config, model_args, output_embedding_size=192):
        self.model_args = model_args
        self.output_embedding_size = output_embedding_size
        if hasattr(config, "output_embedding_size"):
            #优先使用config文件里的输出维度
            self.output_embedding_size = config.output_embedding_size
        elif model_args is not None and model_args.output_embedding_size is not None:
            # 如果config里没有输出维度，在输入参数中指定
            self.output_embedding_size = model_args.output_embedding_size
        else:
            raise NotImplementedError("Please specify output_embedding_size in config or model augments.")
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        if hasattr(config, "embedding_operation") and 'mean-pooling' in config.embedding_operation:
            self.activation = ACT2FN['mish']
            self.my_op = self.mean_pooling
        else:
            self.norm = nn.LayerNorm(self.output_embedding_size)
            self.my_op = self.linear_operation
    def _embed_op(self, outputs, attention_mask=None):
        return self.my_op(outputs, attention_mask)
    def mean_pooling(self, outputs, attention_mask):
        token_embeddings = self.activation(outputs.last_hidden_state)
        token_embeddings = self.embeddingHead(token_embeddings)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    def linear_operation(self, outputs, attention_mask=None):
        text_embeds = outputs.last_hidden_state[:, 0]
        #norm is important in Inner Product
        text_embeds = self.norm(self.embeddingHead(text_embeds))
        return text_embeds
#################
from pytorch_quantization.nn import QuantLinear, TensorQuantizer
class QuantLinearLayer(QuantLinear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinearLayer, self).__init__(in_features, out_features, bias, **kwargs)
        if bias:
            self._aftergemm_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
        self._output_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
    def forward(self, input, output_int8=False):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)
        #output = F.linear(quant_input, quant_weight, bias=self.bias)
        if self.bias is not None:
            output = self.bias + self._aftergemm_quantizer(F.linear(quant_input, quant_weight, None))
        else:
            #default it is this line
            output = F.linear(quant_input, quant_weight, bias=self.bias)
        #The above is all fake-quantized. float_input_1 = float_input * scale / scale
        if output_int8:
            self._output_input_quantizer._fake_quant = False
        else:
            self._output_input_quantizer._fake_quant = True
        output = self._output_input_quantizer(output)
        return output
class QuatOperation(EmbeddingOperation):
    def __init__(self, config, model_args, output_embedding_size=192):
        self.model_args = model_args
        self.output_embedding_size = output_embedding_size
        if hasattr(config, "output_embedding_size"):
            # 优先使用config文件里的输出维度
            self.output_embedding_size = config.output_embedding_size
        elif model_args is not None and model_args.output_embedding_size is not None:
            # 如果config里没有输出维度，在输入参数中指定
            self.output_embedding_size = model_args.output_embedding_size
        else:
            raise NotImplementedError("Please specify output_embedding_size in config or model augments.")
        self.embeddingHead = QuantLinearLayer(config.hidden_size, self.output_embedding_size)
        self.activation = ACT2FN['mish']
    def _embed_op(self, outputs, attention_mask=None, output_int8=False):
        return self.mean_pooling(outputs, attention_mask, output_int8)
    def mean_pooling(self, outputs, attention_mask, output_int8=False):
        token_embeddings = self.activation(outputs.last_hidden_state)
        token_embeddings = self.embeddingHead(token_embeddings, output_int8)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
##############
# 继承自己实现的向量操作类即可：例如LinearOperation
#############
class BertDense(BertPreTrainedModel, GeneralOperation):
    def __init__(self, config, model_args=None):
        BertPreTrainedModel.__init__(self, config)
        GeneralOperation.__init__(self, config, model_args)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.apply(self._init_weights)
    def forward(self, input_ids, attention_mask, return_dict=False):
        outputs = self.bert(input_ids, attention_mask, return_dict=True)
        text_embeds = self._embed_op(outputs, attention_mask)
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds
class QuatBertDense(BertPreTrainedModel, QuatOperation):
    def __init__(self, config, model_args=None):
        BertPreTrainedModel.__init__(self, config)
        QuatOperation.__init__(self, config, model_args)
        self.bert = BertModel(config, add_pooling_layer=False)
        self._final_input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
        self.apply(self._init_weights)
    def forward(self, input_ids, attention_mask, return_dict=False, output_int8=False):
        outputs = self.bert(input_ids, attention_mask, return_dict=True)
        if output_int8 and self._final_input_quantizer._disabled:
            text_embeds = self._embed_op(outputs, attention_mask, output_int8)
        else:
            text_embeds = self._embed_op(outputs, attention_mask)#float
        if output_int8 and not self._final_input_quantizer._disabled:
            self._final_input_quantizer._fake_quant = False
        else:
            self._final_input_quantizer._fake_quant = True
        #_disabled takes the first priority
        text_embeds = self._final_input_quantizer(text_embeds)
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds
'''
#This class is the same as the online training version, but it is wrong.
class QuatBertDense(BertPreTrainedModel, QuatOperation):
    def __init__(self, config, model_args=None):
        BertPreTrainedModel.__init__(self, config)
        QuatOperation.__init__(self, config, model_args)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.apply(self._init_weights)
    def forward(self, input_ids, attention_mask, return_dict=False, output_int8=False):
        outputs = self.bert(input_ids, attention_mask, return_dict=True)
        text_embeds = self._embed_op(outputs, attention_mask, output_int8)
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds
'''
class RobertaDense(RobertaPreTrainedModel):
    # TODO to be implemented
    def __init__(self, config):
        RobertaPreTrainedModel.__init__(self, config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
    def forward(self, input_ids, attention_mask, return_dict=False):
        outputs = self.roberta(input_ids, attention_mask, return_dict=True)
        text_embeds = outputs.last_hidden_state[:, 0]
        if self.config.similarity_metric == "METRIC_IP":
            pass
        elif self.config.similarity_metric == "METRIC_COS":
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        else:
            raise NotImplementedError()
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds
# -> Union[BertDense, RobertaDense, QuatBertDense]
def dense_from_pretrained(model_path: str, model_args):
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type == "bert":
        if 'quat' in config.embedding_operation:
            #config = AutoConfig.from_pretrained(model_path,embedding_operation='quat-mean-pooling')
            model = QuatBertDense.from_pretrained(model_path, config=config, model_args=model_args)
        else:
            model = BertDense.from_pretrained(model_path, config=config, model_args=model_args)
    elif config.model_type == "roberta":
        model = RobertaDense.from_pretrained(model_path, config=config)
    else:
        raise NotImplementedError()
    return model
