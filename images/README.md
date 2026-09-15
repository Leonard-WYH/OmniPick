# Docker 镜像

本目录用于存放 OmniPick 的两个运行镜像归档。由于文件体积较大，`*.tar` 已被
`.gitignore` 排除，请从
[Hugging Face](https://huggingface.co/datasets/Leonard-WYH/OmniPick-Docker-Image/tree/main)
下载。

```bash
cd ~/OmniPick
hf download Leonard-WYH/OmniPick-Docker-Image \
  supermarket_sorting_server.tar \
  supermarket_sorting_client-nav2.tar \
  --repo-type dataset \
  --local-dir images
```

加载并检查镜像：

```bash
docker load -i images/supermarket_sorting_server.tar
docker load -i images/supermarket_sorting_client-nav2.tar
docker image ls | grep supermarket_sorting
```

应得到以下标签：

- `supermarket_sorting:server`
- `supermarket_sorting:client-nav2`

镜像运行需要 NVIDIA 驱动、Docker、NVIDIA Container Toolkit 和可用的 X11 显示。
完整启动方法见 [项目 README](../README.md)。
