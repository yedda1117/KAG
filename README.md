1. 云端部署ragflow
（1）更改 vm.max_map_count 值，设定为：=262144
      sudo sysctl -w vm.max_map_count=262144

（2）在数据持久化目录下克隆仓库
     git clone https://github.com/infiniflow/ragflow.git

（3）进入docker 文件夹，编辑docker-compose-base.yml配置文件中的数据路径为自己的云服务器的持久化路径

（4）在docker-compose-base.yml配置文件中，新增ollama容器配置如下：

     ragflow-ollama:
    image: ollama/ollama
    ports:
      - "11435:11434"
    volumes:
      - /home/featurize/work/ollama:/root/.ollama
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

（5）利用GPU方式拉取docker
      docker compose -f docker-compose-gpu up -d

（6）拉取完毕后 利用docker ps检验容器运行情况

2. 云端拉取模型

（1）进入自定义的ollama容器
   docker exec -it [自定义ollama容器名称]

（2）拉取相关模型
   pull qwen2.5:14b

（3）检验模型拉取情况
  docker exec -it [自定义ollama容器名称] ollama list

3. ragflow网络服务配置

（1）将ragflow-serve端口、自定义的ollama容器映射到外网，根据对应地址在app.py的基础配置中做对应更改
   （根据云服务器具体情况而言，若不需要则直接按照docker ps显示端口，直接在浏览器打开）

（2）在ragflow中进行模型的相关系统配置

4. 聊天交互配置

（1）将知识库.xlsx注入ragflow数据端，实现解析

（2）建立聊天助手，选择相关知识库以及chat模型，将更新后的chat_id在app.py的基础配置中做对应更改

5. 系统运行

（1）做完以上基础配置后，运行main.py

（2）点击返回内容中地址，即可在浏览器上进行系统的使用



    
