# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser, BitsAndBytesConfig

from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config,global_args
from models import MyTransformer, Generate,BaiChuanConfig,BaiChuanTokenizer

deep_config = get_deepspeed_config()

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config(config_class_name=BaiChuanConfig,
                                                                 tokenizer_class_name=BaiChuanTokenizer)
    config.pad_token_id = config.eos_token_id

    load_in_4bit = False
    if not load_in_4bit:
        load_in_4bit_config = {}
    else:
        load_in_4bit_config = dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            load_in_4bit=True,
            # device_map="auto",
            device_map={"": 0}  # 第一块卡
        )

    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=torch.float16,
                             **load_in_4bit_config,)

    model = pl_model.get_llm_model()
    model = model.eval()
    model.requires_grad_(False)

    if not load_in_4bit:
        model = model.half()
    model.cuda()

    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 "登鹳雀楼->王之涣\n夜雨寄北->",
                 "Hamlet->Shakespeare\nOne Hundred Years of Solitude->",
                 ]
    for input in text_list:
        response, history = Generate.generate(model, query=input, tokenizer=tokenizer, max_length=512,
                                          eos_token_id=config.eos_token_id,pad_token_id=config.eos_token_id,
                                          do_sample=False, top_p=0.7, temperature=0.95, )
        print('input', input)
        print('output', response)