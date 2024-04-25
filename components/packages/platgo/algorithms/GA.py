import numpy as np

from components.packages.platgo.GeneticAlgorithm import GeneticAlgorithm
from components.packages.platgo.utils.fitness_single import fitness_single
from components.packages.platgo.utils.selections.tournament_selection import (
    tournament_selection,
)  # noqa
from components.packages.platgo.operators.OperatorGA import OperatorGA

import torch
import os
import numpy as np
import re
import transformers

from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)


class GA(GeneticAlgorithm):
    type = {
        "n_obj": "single",
        "encoding": {"real", "binary", "permutation"},
        "special": {"large/none", "constrained/none"},
    }

    def __init__(
        self,
        pop_size,
        options,
        optimization_problem,
        control_cb,
        max_fe=10000,
        name="GA",
        show_bar=False,
        sim_req_cb=None,
        ext_opt_prob_cb=None,
        debug=False,
        proC=1,
        disC=20,
        proM=1,
        disM=20,
    ) -> None:

        super(GA, self).__init__(
            pop_size,
            options,
            optimization_problem,
            control_cb,
            max_fe=max_fe,
            name=name,
            show_bar=show_bar,
            sim_req_cb=sim_req_cb,
            ext_opt_prob_cb=ext_opt_prob_cb,
            debug=debug,
        )
        self.proC = proC
        self.disC = disC
        self.proM = proM
        self.disM = disM

    def run_algorithm(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        #
        # # 导入模型
        # pretrained_model_path = 'C:\\Users\\Ye\\.cache\\modelscope\\hub\\shakechen\\Llama-2-7b-chat-hf'
        # # pretrained_model_path = 'D:\Github_repository\Llama-2-7b-chat-hf'
        #
        # # 1. 使用 AutoConfig.from_pretrained() 方法加载预训练模型的配置
        # config = AutoConfig.from_pretrained(
        #     # max_length=1024,
        #     pretrained_model_name_or_path=pretrained_model_path,
        # )
        # # 此处修改默认配置，开启增量推理能够加速推理性能.增量推理是一种技术，可以在生成文本时利用先前的计算结果，从而提高生成效率。
        # config.use_past = True
        #
        # # 2. 加载权重
        # # 创建 BitsAndBytesConfig 对象，用于配置模型的量化参数。这些参数用于控制模型的权重加载方式和计算精度。
        # quantization_config = BitsAndBytesConfig(
        #     load_in_bits=8,  # 8 位
        #     load_in_bytes=1,  # 1 字节
        #     bnb_4bit_compute_dtype=torch.float16  # 设置为 torch.float16
        # )
        # # 使用 AutoModelForCausalLM.from_pretrained() 方法加载预训练模型的权重，指定了模型配置、设备映射、量化配置等参数
        # model = AutoModelForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path=pretrained_model_path,
        #     config=config,
        #     device_map='auto',
        #     quantization_config=quantization_config
        # )
        #
        # # 3. 使用 AutoTokenizer.from_pretrained() 方法加载分词器，将输入转化为模型可接受数据
        # tokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path=pretrained_model_path,
        # )
        #
        # # 4. Define the pipeline of settings for the task  定义对于任务的管道设置
        # pipeline = transformers.pipeline(
        #     "text-generation",  # 指定了任务类型为 "text-generation"
        #     model=model,  # 模型为加载的模型
        #     tokenizer=tokenizer,  # 分词器为加载的分词器
        # )

        pop = self.problem.init_pop()
        self.cal_obj(pop)
        while self.not_terminal(pop):
            MatingPool = tournament_selection(
                2, self.problem.pop_size, fitness_single(pop)
            )  # noqa
            Offspring = OperatorGA(
                pop[MatingPool],
                self.problem,
                self.proC,
                self.disC,
                self.proM,
                self.disM,

            )  # noqa

            self.cal_obj(Offspring)
            pop = pop + Offspring
            rank = np.argsort(fitness_single(pop), kind="mergesort")
            pop = pop[rank[0 : self.problem.pop_size]]
            print(np.min(pop.objv))
        return pop
