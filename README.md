# model_factory 对 LLaMA-Factory 的一次重构和简化
主要用于对 chatglm, internlm, llama 等模型的微调（SFT），其底层基于 hugging-face 的 Trainer，分布式性能优化有两种可选：
1、hugging-face 的 accelerate
2、DeepSpeed
他们通过配置文件生效，也可以通过对应的启动工具启动，配置文件在 config/ 目录下。
