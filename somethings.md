## paper to read

https://www.usenix.org/system/files/osdi24_slides-lee.pdf infinegen
https://zhuanlan.zhihu.com/p/2024601794567762221 dualpath

写一个 费曼学习法 skill


快速 bash 行首 行尾

写一下 自己对于ai的分析 是作为consultant（it桔子 这种类似的 可以给人来带每日资讯的 不过真的有用吗？），或者一个管家

从硬件路径看 Tensor 搬运 补充一个blog ref：https://zhuanlan.zhihu.com/p/2042997288373589257


https://stock-sdk.linkdiary.cn/guide/getting-started

写作的目的不在于写完，而在于增进你自己的理解，进而增进周围人的理解。

让 AI 为你写作，就像花钱请人为你健身一样。


作者：葛志勇
链接：https://zhuanlan.zhihu.com/p/2046254612638967638
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



https://hebiao064.github.io/rl-weight-sync

https://forceinjection.github.io/01_hardware_architecture/performance/04_pcie_domain_numa.html


https://github.com/stonet-research/cheops25-IO-characterization-of-LLM-model-kv-cache-offloading-nvme/blob/main/figure5-6-kv-offloading-flexgen/flexgen-opt-6.7b-kv-trace.py   io 时间test


https://www.feishu.cn/drive/home?from=jiancun


https://github.com/fzyzcjy/torch_memory_saver
写成实习项目

我们支持使用原生 vLLM 卸载和Mooncake将 KV 缓存分层卸载到 CPU 和磁盘。更大的 KV 缓存空间可以提高并发性，从而分摊更多的训练器成本。

这些方法之间主要有两点不同：

vLLM 原生卸载——一种简单的方法，为每个工作进程（DP 进程）创建一个 CPU/磁盘池；只有该工作进程才能从该缓存加载数据。
Mooncake Store则作为集中式存储运行，它将所有客户端（节点）的 RAM/磁盘汇集到一个大池中，然后任何节点上的任何推理工作进程都可以访问该池——这提供了显著的优势，尤其是在使用更复杂的路由策略时。


看一下 原生的怎么做的 kvcahe 怎么做的  lmcache？ llmd？


https://winterresearch.com/big_debt_crises


https://github.com/0voice/kernel_new_features/blob/main/cgroups/%E6%96%87%E7%AB%A0/Cgroup%20-%20Linux%E7%9A%84IO%E8%B5%84%E6%BA%90%E9%9A%94%E7%A6%BB.md
Linux 内核

应该写一个全景计算和说明的。省掉的计算时间 = 计算量 / GPU 计算吞吐

传输时间 = 要传输的 kvcache 大小 / kvcache 传输带宽 B = min(B_h2d, B_nic) 比如 到底带宽。吞吐等等



我记得要看知乎的一个文章 说magatron-fsdp的



https://yifanqiao.notion.site/Solve-the-GPU-Cost-Crisis-with-kvcached-289da9d1f4d68034b17bf2774201b141

快速序列化 和反序列化到底怎么说

经典的数字
比如 带宽什么的

https://github.com/doongz/notes


https://guanjiawei.ai/zh/blog/goal-two-personalities



https://forceinjection.github.io/09_inference_system/kv_cache/02_systems/lmcache/local_cpu_backend.html




看看别人的infra 怎么做的
[cosmos ](https://github.com/NVIDIA/cosmos-framework/tree/main/cosmos_framework/model) 里面有文档



https://www.1point3acres.com/home/pins/1180983

