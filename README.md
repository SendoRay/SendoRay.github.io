# chengzhy's Blog

This is my personal blog built with [Hugo](https://gohugo.io/) and [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

## paper to read

https://www.usenix.org/system/files/osdi24_slides-lee.pdf infinegen
https://zhuanlan.zhihu.com/p/2024601794567762221 dualpath

写一个 费曼学习法 skill


```bash

# 进项目根目录
cd /Users/chengzhy/SendoRay.github.io

# 1.1) 普通本地预览（不显示 draft）
hugo server

# 1.2) 包含 draft / 未来日期文章（你这篇 06-11 用得上）
hugo server -D -F

# 2) 看当前的tags 不要乱加新的tags
make tags

```

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




应该写一个全景计算和说明的。省掉的计算时间 = 计算量 / GPU 计算吞吐

传输时间 = 要传输的 kvcache 大小 / kvcache 传输带宽 B = min(B_h2d, B_nic) 比如 到底带宽。吞吐等等



我记得要看知乎的一个文章 说magatron-fsdp的





快速序列化 和反序列化到底怎么说

经典的数字
比如 带宽什么的


https://guanjiawei.ai/zh/blog/goal-two-personalities


Kubernetes


看看别人的infra 怎么做的
[cosmos ](https://github.com/NVIDIA/cosmos-framework/tree/main/cosmos_framework/model) 里面有文档