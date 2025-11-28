FROM micr.cloud.mioffice.cn/chenzhiyang1/gumiho:v1

# 避免 torch 被覆盖
# COPY requirements.txt /tmp/requirements.txt
# RUN sed -i '/torch/d' /tmp/requirements.txt
# RUN sed -i '/pytorch-triton/d' /tmp/requirements.txt
# RUN sed -i '/rocm/d' /tmp/requirements.txt
# RUN sed -i '/cmake/d' /tmp/requirements.txt
# RUN sed -i '/protobuf/d' /tmp/requirements.txt
# RUN sed -i '/tensorboard/d' /tmp/requirements.txt

# RUN pip install -r /tmp/requirements.txt

RUN pip install addict accelerate deepspeed wandb shortuuid fschat openai anthropic

# 5. 设置工作目录（容器启动进入）
WORKDIR /workspace

# 6. 默认挂载点（可不写）
VOLUME ["/workspace"]
