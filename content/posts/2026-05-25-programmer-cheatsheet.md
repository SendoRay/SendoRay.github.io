---
title: 程序员速查手册
date: '2026-05-25'
tags:
- Engineering

draft: false
ShowToc: true
TocOpen: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
---

> 一份覆盖日常开发高频场景的速查手册，涵盖 Git 规范、Linux 命令、Vim、Shell 脚本、Docker、正则表达式、tmux、行业黑话。

---

## 目录

1. [Git 命名与提交规范](#1-git-命名与提交规范)
2. [Git 常用命令速查](#2-git-常用命令速查)
3. [Linux 常用命令](#3-linux-常用命令)
4. [Vim 常用快捷键](#4-vim-常用快捷键)
5. [Shell / Bash 脚本语法](#5-shell--bash-脚本语法)
6. [Docker 常用命令](#6-docker-常用命令)
7. [正则表达式速查](#7-正则表达式速查)
8. [tmux 命令行管理](#8-tmux-命令行管理)
9. [行业黑话速查](#9-行业黑话速查)
10. [优质在线资源推荐](#10-优质在线资源推荐)

---

## 1. Git 命名与提交规范

### 1.1 分支命名规范

主流团队推荐采用「类型/简短描述」的格式，使用小写英文 + 连字符（kebab-case），避免中文、空格、下划线。

| 分支类型 | 命名格式 | 示例 | 说明 |
|---------|---------|------|------|
| 主分支 | `main` / `master` | `main` | 永远保持可发布状态 |
| 开发主线 | `develop` / `dev` | `develop` | 日常集成分支 |
| 新功能 | `feature/<名称>` | `feature/user-login` | 从 develop 切出，完成后合回 |
| 缺陷修复 | `fix/<名称>` 或 `bugfix/<名称>` | `fix/login-timeout` | 修复非紧急 bug |
| 紧急修复 | `hotfix/<名称>` | `hotfix/payment-crash` | 直接从 main 切出 |
| 发布准备 | `release/<版本号>` | `release/1.2.0` | 预发布分支 |
| 文档变更 | `docs/<名称>` | `docs/api-readme` | 仅文档改动 |
| 重构 | `refactor/<名称>` | `refactor/order-service` | 不改变功能的代码重构 |
| 实验性 | `experiment/<名称>` | `experiment/new-cache` | 试验性方案 |
| 个人分支 | `<用户名>/<名称>` | `zhangsan/poc-redis` | 不进主仓的临时分支 |

命名建议：

- 全部小写，单词之间用 `-` 连接
- 名称简短表意，不超过 30 个字符
- 可附加 issue 编号，如 `feature/JIRA-123-user-login`
- 不要使用 `tmp`、`test`、`new` 这种无信息量的名字

### 1.2 Commit Message 规范（Conventional Commits）

Conventional Commits 是目前业界最通用的提交规范，被 Angular、Vue、Element 等项目采用。它的格式如下：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型一览：**

| type | 含义 | 使用场景 |
|------|------|---------|
| `feat` | 新功能 | 用户可见的新增功能 |
| `fix` | 修复 bug | 修复缺陷 |
| `docs` | 文档变更 | README、注释、JSDoc 等 |
| `style` | 代码格式 | 不影响逻辑的格式调整（空格、分号、缩进） |
| `refactor` | 重构 | 既不是新功能也不是 bug 修复 |
| `perf` | 性能优化 | 提升性能的修改 |
| `test` | 测试相关 | 新增/修改测试用例 |
| `build` | 构建系统 | 影响构建工具或外部依赖（webpack、npm、maven） |
| `ci` | CI 配置 | GitHub Actions、Jenkins 等 |
| `chore` | 杂项 | 不修改 src 或测试的其他变更 |
| `revert` | 回滚 | 回滚某个 commit |

**示例：**

```
feat(user): 新增手机号登录入口

- 支持验证码登录
- 兼容老版本 token

Closes #1024
```

```
fix(payment): 修复重复扣款问题

并发请求未加分布式锁，导致同一订单可能扣款两次。
现已使用 Redis 分布式锁兜底。

BREAKING CHANGE: 支付接口签名调整，调用方需升级 SDK 至 2.x
```

**写作要点：**

- subject 用动词开头，使用现在时（"add" 而不是 "added"），结尾不加句号
- subject 长度不超过 50 字符，整体首行不超过 72 字符
- body 解释「为什么」而不是「做了什么」
- 涉及破坏性变更，必须在 footer 写 `BREAKING CHANGE:`
- 关联 issue 用 `Closes #123` / `Refs #123`

### 1.3 Tag / 版本号规范（SemVer）

语义化版本：`MAJOR.MINOR.PATCH`，例如 `v1.4.2`。

- `MAJOR`：不兼容的 API 变更
- `MINOR`：向后兼容的新功能
- `PATCH`：向后兼容的 bug 修复
- 预发布：`v1.4.0-beta.1`、`v1.4.0-rc.2`

---

## 2. Git 常用命令速查

### 2.1 基础操作

```bash
git init                          # 初始化仓库
git clone <url>                   # 克隆仓库
git clone <url> --depth=1         # 浅克隆，加快速度
git status                        # 查看状态
git status -s                     # 简洁模式
git add <file>                    # 添加到暂存区
git add -p                        # 交互式分块添加（强烈推荐）
git commit -m "msg"               # 提交
git commit --amend                # 修改最后一次提交（未推送时使用）
git commit --amend --no-edit      # 追加文件到上次提交，不改 message
```

### 2.2 分支管理

```bash
git branch                        # 查看本地分支
git branch -a                     # 查看所有分支（含远程）
git branch -vv                    # 查看分支及其追踪关系
git checkout -b feature/xxx       # 创建并切换分支
git switch -c feature/xxx         # 同上（新版推荐）
git switch main                   # 切换分支
git branch -d <branch>            # 删除分支（已合并）
git branch -D <branch>            # 强制删除
git push origin --delete <branch> # 删除远程分支
git branch -m old new             # 重命名分支
```

### 2.3 同步与合并

```bash
git fetch                         # 拉取远程更新但不合并
git fetch --prune                 # 同时清理已删除的远程分支
git pull                          # fetch + merge
git pull --rebase                 # fetch + rebase（保持线性历史）
git merge <branch>                # 合并分支
git merge --no-ff <branch>        # 强制生成 merge commit
git rebase <branch>               # 变基
git rebase -i HEAD~3              # 交互式变基（合并/编辑最近3个提交）
git cherry-pick <commit>          # 摘取某个提交
```

### 2.4 撤销与回滚

```bash
git restore <file>                # 撤销工作区改动（新版推荐）
git restore --staged <file>       # 取消暂存
git checkout -- <file>            # 旧版撤销工作区改动
git reset HEAD <file>             # 旧版取消暂存
git reset --soft HEAD~1           # 撤销提交，保留改动在暂存区
git reset --mixed HEAD~1          # 撤销提交，保留改动在工作区（默认）
git reset --hard HEAD~1           # 彻底撤销，慎用
git revert <commit>               # 反向提交（用于已推送的提交）
git reflog                        # 查看所有 HEAD 移动记录（救命神器）
```

### 2.5 储藏与暂存

```bash
git stash                         # 储藏当前改动
git stash push -m "msg"           # 带注释储藏
git stash list                    # 查看储藏列表
git stash pop                     # 弹出最近一次储藏
git stash apply stash@{1}         # 应用指定储藏（不删除）
git stash drop stash@{0}          # 删除某条储藏
git stash clear                   # 清空所有储藏
```

### 2.6 远程与协作

```bash
git remote -v                     # 查看远程仓库
git remote add origin <url>       # 添加远程仓库
git push -u origin main           # 推送并设置上游
git push origin --tags            # 推送所有 tag
git push origin :feature/x        # 删除远程分支（旧式）
git tag v1.0.0                    # 创建轻量 tag
git tag -a v1.0.0 -m "release"    # 创建附注 tag
```

### 2.7 查看历史

```bash
git log --oneline --graph --all   # 图形化查看历史（强推）
git log -p <file>                 # 查看文件每次提交的 diff
git log --author="名字"           # 按作者过滤
git log --since="2 weeks ago"     # 按时间过滤
git blame <file>                  # 查看每行最后由谁修改
git diff                          # 工作区 vs 暂存区
git diff --cached                 # 暂存区 vs HEAD
git diff main..feature            # 两个分支差异
git show <commit>                 # 查看某次提交详情
```

### 2.8 救命三连

```bash
git reflog                        # 找回任何丢失的提交
git fsck --lost-found             # 找回悬空对象
git checkout -b rescue <commit>   # 用旧 commit 创建分支救回
```

---

## 3. Linux 常用命令

### 3.1 文件与目录

```bash
ls -alh                  # 列出所有文件，含隐藏，人类可读大小
ll                       # ls -l 别名
pwd                      # 当前路径
cd -                     # 回到上一个路径
cd ~                     # 回到家目录
mkdir -p a/b/c           # 递归创建目录
rm -rf <dir>             # 强制递归删除（务必小心）
cp -r src dst            # 递归复制目录
mv old new               # 移动 / 重命名
ln -s target link        # 创建软链接
touch file               # 创建空文件 / 更新时间戳
stat file                # 查看文件详细信息
file <file>              # 查看文件类型
tree -L 2                # 树形结构，深度2
```

### 3.2 文件查看

```bash
cat file                 # 输出全部内容
tac file                 # 倒序输出
less file                # 分页查看（推荐，q 退出）
head -n 20 file          # 前 20 行
tail -n 20 file          # 后 20 行
tail -f file             # 实时追踪（看日志）
tail -F file             # 文件被轮转时也能追上
wc -l file               # 行数
nl file                  # 显示行号
```

### 3.3 查找

```bash
find . -name "*.log"                   # 按名查找
find . -type f -size +100M             # 大于 100M 的文件
find . -mtime -7                       # 7 天内修改过
find . -name "*.tmp" -delete           # 找到并删除
find . -name "*.py" -exec grep -l "TODO" {} \;
locate filename                        # 数据库查找（需 updatedb）
which python                           # 查命令路径
whereis python                         # 查二进制/man/源码位置
```

### 3.4 文本处理

```bash
grep "pattern" file                    # 基本搜索
grep -r "pattern" .                    # 递归搜索
grep -i "pattern" file                 # 忽略大小写
grep -v "pattern" file                 # 反向匹配
grep -n "pattern" file                 # 显示行号
grep -E "a|b" file                     # 扩展正则
grep -A 3 -B 2 "err" log               # 匹配前2行后3行

sed 's/old/new/g' file                 # 替换所有
sed -i 's/old/new/g' file              # 原地修改
sed -n '10,20p' file                   # 打印 10~20 行
sed '/^$/d' file                       # 删空行

awk '{print $1}' file                  # 第一列
awk -F: '{print $1}' /etc/passwd       # 指定分隔符
awk '$3 > 100 {print $0}' file         # 条件过滤
awk 'NR==FNR{a[$1]; next} $1 in a' a b # 取交集

sort file                              # 排序
sort -u file                           # 去重排序
sort -k2 -n file                       # 按第二列数值排序
uniq -c file                           # 统计重复次数（需先排序）
cut -d: -f1 /etc/passwd                # 按分隔符取列
tr 'a-z' 'A-Z' < file                  # 字符替换
xargs                                  # 把 stdin 当参数（详见管道章节）
```

### 3.5 权限与用户

```bash
chmod 755 file                # rwxr-xr-x
chmod +x file                 # 添加可执行
chmod -R 644 dir              # 递归
chown user:group file         # 修改归属
sudo -i                       # 切换 root
su - username                 # 切换用户并加载环境
id                            # 当前用户的 uid/gid
groups <user>                 # 用户所在组
passwd                        # 修改自己密码
```

权限数字含义：r=4, w=2, x=1。755 = rwx (owner) + r-x (group) + r-x (other)。

### 3.6 进程与系统

```bash
ps aux                        # 所有进程
ps -ef | grep nginx           # 查找进程
top                           # 实时进程（推荐 htop）
htop                          # 增强版（需安装）
kill <pid>                    # 优雅结束
kill -9 <pid>                 # 强制杀死
pkill -f "java.*MyApp"        # 按名字模式杀
pgrep -f nginx                # 按名字找 pid
nohup cmd &                   # 后台运行，不挂断
jobs                          # 当前 shell 后台任务
fg %1                         # 调到前台
bg %1                         # 后台继续
disown -h %1                  # 脱离当前 shell

uptime                        # 系统运行时间 + 负载
free -h                       # 内存使用
df -h                         # 磁盘使用
du -sh *                      # 当前目录每个项目大小
du -sh * | sort -hr | head    # 找最大的目录
lsof -i:8080                  # 谁在占用 8080 端口
lsof -p <pid>                 # 进程打开的文件
```

### 3.7 网络

```bash
ip addr                       # 网卡信息（替代 ifconfig）
ip route                      # 路由表
ping host                     # 联通性
traceroute host               # 路由跳点
mtr host                      # ping + traceroute 合体
curl -I https://example.com   # 仅获取 HTTP 头
curl -X POST -d "k=v" url     # POST 请求
curl -O url                   # 下载并保留文件名
wget url                      # 下载
ss -tnlp                      # 监听中的 TCP 端口（替代 netstat）
ss -tan                       # 所有 TCP 连接
nc -zv host 22                # 探测端口
dig example.com               # DNS 查询
nslookup example.com          # DNS 查询
scp file user@host:/path      # 远程拷贝
rsync -avz src/ user@host:dst/ # 增量同步（强推）
ssh user@host                 # 远程登录
ssh -i key.pem user@host      # 指定密钥
```

### 3.8 压缩归档

```bash
tar -czvf x.tar.gz dir/       # 打包并 gzip 压缩
tar -xzvf x.tar.gz            # 解压
tar -xzvf x.tar.gz -C /opt/   # 解压到指定目录
tar -tzvf x.tar.gz            # 仅查看不解压
zip -r x.zip dir/             # zip 打包
unzip x.zip -d target/        # zip 解压
gzip file                     # 单文件压缩
gunzip file.gz                # 解压
```

记忆口诀：`czf` 创建，`xzf` 解压，`tzf` 查看，`v` 显示过程。

### 3.9 管道与重定向

```bash
cmd > file                    # 标准输出覆盖写
cmd >> file                   # 追加
cmd 2> err.log                # 错误流写入
cmd > out.log 2>&1            # 全部写入同一文件
cmd &> all.log                # 同上简写（bash）
cmd < input.txt               # 从文件读输入
cmd1 | cmd2                   # 管道
cmd1 | tee file | cmd2        # 同时落盘 + 继续管道
```

xargs 高频用法：

```bash
# 批量删除找到的文件
find . -name "*.tmp" | xargs rm

# 文件名含空格时
find . -name "*.tmp" -print0 | xargs -0 rm

# 并行 4 个进程处理
cat urls.txt | xargs -P 4 -n 1 wget
```

### 3.10 SSH 配置（提效神技）

`~/.ssh/config`：

```
Host myserver
    HostName 192.168.1.100
    User root
    Port 22
    IdentityFile ~/.ssh/id_rsa

Host *.alibaba-inc.com
    User chengzhiyuan.czy
    ForwardAgent yes
```

之后只需 `ssh myserver` 即可登录。

---

## 4. Vim 常用快捷键

### 4.1 模式切换

| 按键 | 含义 |
|------|------|
| `Esc` | 回到普通模式 |
| `i` / `a` | 进入插入模式（光标前/后） |
| `I` / `A` | 行首/行尾插入 |
| `o` / `O` | 下方/上方新建行并插入 |
| `v` | 字符可视模式 |
| `V` | 行可视模式 |
| `Ctrl+v` | 块可视模式（列选择神器） |
| `:` | 命令模式 |

### 4.2 移动

| 按键 | 含义 |
|------|------|
| `h j k l` | 左/下/上/右 |
| `w` / `b` | 下一个/上一个单词 |
| `0` / `^` | 行首（含空格/不含空格） |
| `$` | 行尾 |
| `gg` / `G` | 文件首/尾 |
| `:n` | 跳到第 n 行 |
| `Ctrl+u` / `Ctrl+d` | 上翻/下翻半页 |
| `Ctrl+f` / `Ctrl+b` | 翻页 |
| `%` | 匹配的括号 |
| `*` | 搜索光标处单词 |

### 4.3 编辑

| 按键 | 含义 |
|------|------|
| `x` | 删除字符 |
| `dd` | 删除整行 |
| `dw` | 删除一个单词 |
| `d$` / `D` | 删除到行尾 |
| `yy` | 复制整行 |
| `yw` | 复制单词 |
| `p` / `P` | 粘贴到光标后/前 |
| `u` | 撤销 |
| `Ctrl+r` | 重做 |
| `r<char>` | 替换单个字符 |
| `cc` | 删除整行并进入插入 |
| `ciw` | 删除光标所在单词并插入 |
| `ci"` | 删除引号内内容并插入（神技） |
| `>>` / `<<` | 增加/减少缩进 |

### 4.4 搜索替换

```
/pattern        # 向后搜索
?pattern        # 向前搜索
n / N           # 下一个 / 上一个
:%s/old/new/g   # 全文替换
:%s/old/new/gc  # 全文替换（每处确认）
:5,20s/old/new/g # 5~20 行替换
```

### 4.5 文件操作

```
:w              # 保存
:q              # 退出
:wq / :x / ZZ   # 保存退出
:q!             # 不保存退出
:e file         # 打开文件
:sp file        # 水平分屏打开
:vsp file       # 垂直分屏
Ctrl+w + h/j/k/l  # 切换窗口
```

---

## 5. Shell / Bash 脚本语法

### 5.1 脚本头与变量

```bash
#!/usr/bin/env bash
set -euo pipefail   # 推荐：出错即停 / 未定义变量报错 / 管道任一失败即失败

NAME="world"        # 赋值不能有空格
echo "Hello $NAME"
echo "Hello ${NAME}!"
readonly PI=3.14    # 常量
unset NAME          # 删除变量
```

### 5.2 字符串操作

```bash
str="hello world"
echo ${#str}              # 长度
echo ${str:0:5}           # 子串：hello
echo ${str/world/bash}    # 替换：hello bash
echo ${str^^}             # 转大写
echo ${str,,}             # 转小写

# 默认值
echo ${VAR:-default}      # VAR 未设置时使用 default
echo ${VAR:=default}      # 未设置则同时赋值
echo ${VAR:?error msg}    # 未设置则报错退出
```

### 5.3 条件判断

```bash
# 数值比较：-eq -ne -lt -le -gt -ge
if [ "$x" -gt 10 ]; then
    echo "big"
elif [ "$x" -eq 10 ]; then
    echo "ten"
else
    echo "small"
fi

# 字符串：= != -z(空) -n(非空)
if [ -z "$str" ]; then ...; fi

# 文件：-f 普通文件 / -d 目录 / -e 存在 / -r 可读 / -w 可写 / -x 可执行
if [ -f /etc/passwd ]; then ...; fi

# 推荐使用 [[ ]]，支持正则与 && ||
if [[ "$name" =~ ^[A-Z] && -f "$file" ]]; then ...; fi
```

### 5.4 循环

```bash
for i in 1 2 3; do echo $i; done
for i in {1..10}; do echo $i; done
for i in $(seq 1 10); do echo $i; done
for f in *.log; do echo "$f"; done

while read -r line; do
    echo "$line"
done < file.txt

# 经典 C 风格
for ((i=0; i<10; i++)); do
    echo $i
done
```

### 5.5 函数与参数

```bash
greet() {
    local name="$1"          # local 限定作用域
    echo "hello $name"
    return 0
}

greet "Alice"

# 脚本参数
$0    # 脚本名
$1..$9 # 第 n 个参数
$#    # 参数个数
$@    # 所有参数（逐个）
$*    # 所有参数（整体）
$?    # 上一条命令退出码
$$    # 当前进程 PID
$!    # 后台最后进程 PID
```

### 5.6 实用片段

```bash
# 安全的错误处理
trap 'echo "脚本第 $LINENO 行出错"; exit 1' ERR

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 判断命令是否存在
if ! command -v jq &>/dev/null; then
    echo "请先安装 jq"; exit 1
fi

# 读取用户输入
read -rp "请输入名字: " name

# 把命令输出存为数组
mapfile -t files < <(ls *.log)
```

---

## 6. Docker 常用命令

### 6.1 镜像

```bash
docker images                       # 列出镜像
docker pull nginx:1.25              # 拉镜像
docker rmi <image>                  # 删除镜像
docker build -t myapp:1.0 .         # 构建
docker tag myapp:1.0 repo/myapp:1.0 # 打标签
docker push repo/myapp:1.0          # 推送
docker save -o app.tar myapp:1.0    # 导出
docker load -i app.tar              # 导入
docker image prune -a               # 清理无用镜像
```

### 6.2 容器

```bash
docker ps                           # 运行中
docker ps -a                        # 包含已停止
docker run -d --name web -p 80:80 nginx
docker run -it --rm ubuntu bash     # 交互式临时容器
docker run -v /host:/container ...  # 挂载卷
docker run -e ENV=prod ...          # 环境变量
docker exec -it web bash            # 进入运行容器
docker logs -f web                  # 实时日志
docker stop web                     # 停止
docker start web                    # 启动
docker restart web                  # 重启
docker rm web                       # 删除（需先停）
docker rm -f web                    # 强制删除
docker stats                        # 实时资源使用
docker inspect web                  # 详细信息
docker cp web:/path/file ./         # 容器内文件拷出
docker container prune              # 清理已停容器
```

### 6.3 网络与卷

```bash
docker network ls
docker network create mynet
docker run --network=mynet ...

docker volume ls
docker volume create mydata
docker run -v mydata:/data ...
docker volume prune
```

### 6.4 Docker Compose

```bash
docker-compose up -d                # 后台启动
docker-compose down                 # 停止并删除
docker-compose ps                   # 查看状态
docker-compose logs -f web          # 查看日志
docker-compose exec web bash        # 进入容器
docker-compose build --no-cache     # 重建
docker-compose pull                 # 拉镜像
docker-compose restart web          # 重启服务
```

### 6.5 一键清理

```bash
docker system df                    # 看看占了多少空间
docker system prune -a --volumes    # 全清（慎用）
```

---

## 7. 正则表达式速查

### 7.1 字符匹配

| 符号 | 含义 |
|------|------|
| `.` | 任意单字符（不含换行） |
| `\d` | 数字，等同 `[0-9]` |
| `\D` | 非数字 |
| `\w` | 字母/数字/下划线 |
| `\W` | 非 `\w` |
| `\s` | 空白（空格/Tab/换行） |
| `\S` | 非空白 |
| `[abc]` | a 或 b 或 c |
| `[^abc]` | 非 a/b/c |
| `[a-z]` | 小写字母 |
| `\b` | 单词边界 |

### 7.2 量词

| 符号 | 含义 |
|------|------|
| `*` | 0 次或多次 |
| `+` | 1 次或多次 |
| `?` | 0 或 1 次 |
| `{n}` | 恰好 n 次 |
| `{n,}` | 至少 n 次 |
| `{n,m}` | n~m 次 |
| `*?` `+?` | 非贪婪匹配 |

### 7.3 锚点与分组

| 符号 | 含义 |
|------|------|
| `^` | 行首 |
| `$` | 行尾 |
| `(abc)` | 捕获分组 |
| `(?:abc)` | 非捕获分组 |
| `(?=abc)` | 正向先行断言 |
| `(?!abc)` | 负向先行断言 |
| `(?<=abc)` | 正向后行断言 |
| `\1 \2` | 反向引用第 n 个分组 |

### 7.4 常用模式

```regex
# 邮箱
^[\w.+-]+@[\w-]+\.[\w.-]+$

# URL
^https?://[\w.-]+(:\d+)?(/[^\s]*)?$

# 中国大陆手机号
^1[3-9]\d{9}$

# 身份证（18 位）
^\d{17}[\dXx]$

# IPv4
^((25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(25[0-5]|2[0-4]\d|[01]?\d?\d)$

# 日期 YYYY-MM-DD
^\d{4}-\d{2}-\d{2}$

# 中文字符
[\u4e00-\u9fa5]

# 强密码（8+，大小写+数字）
^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$
```

### 7.5 引擎差异提示

- POSIX BRE（grep 默认）：`+ ? | ( )` 需转义为 `\+ \? \| \( \)`
- POSIX ERE（grep -E、egrep）：上述符号无需转义
- PCRE（Perl/Python/JS/Java）：支持先行/后行断言、命名分组等高级语法
- 推荐在线测试：[regex101.com](https://regex101.com)

---

## 8. tmux 命令行管理

tmux 是终端复用器，核心价值：**断开 SSH 后进程不终止**，且支持多窗口/分屏。所有内部快捷键的前缀为 `Ctrl+b`（先按 Ctrl+b 松开，再按功能键）。

### 8.1 会话管理

| 命令 | 作用 |
|------|------|
| `tmux ls` 或 `tmux list-sessions` | 列出所有运行中的会话 |
| `tmux new -s <名称>` | 创建并进入新会话 |
| `tmux attach -t <名称>` | 重新连接指定会话 |
| `tmux kill-server` | 一键关闭所有 tmux 会话及服务（彻底重置） |
| `tmux kill-session -t <名称>` | 关闭指定会话 |
| `tmux detach` 或 `Ctrl+b d` | 脱离当前会话（会话后台继续运行） |
| `tmux rename-session -t <旧名> <新名>` | 重命名会话 |
| `tmux switch -t <名称>` | 切换到另一个会话 |

**加载/恢复旧会话的典型流程：**

```bash
# 场景：SSH 断开后重连
# 1. 先列出正在运行的会话
tmux ls
# 输出示例：
# dev: 3 windows (created Mon May 25 10:00:00 2026)
# train: 1 windows (created Mon May 25 09:30:00 2026)

# 2. 连回指定会话
tmux attach -t dev
# 或简写
tmux a -t dev

# 3. 如果只有一个会话，直接接上
tmux a

# 4. 在会话内脱离（不终止）
# 按 Ctrl+b d

# 5. 如果忘了会话名，接上最近的
tmux a
```

### 8.2 窗口与分屏快捷键

前缀：`Ctrl+b`

| 功能 | 按键顺序 | 说明 |
|------|---------|------|
| 新建窗口 | `c` | `Ctrl+b c` |
| 重命名窗口 | `,` | `Ctrl+b ,` 输入新名称回车 |
| 切换窗口 | `0-9` | `Ctrl+b 0` 切到第 0 个窗口 |
| 上下分屏 | `"` | 水平分割，焦点留在上方 |
| 左右分屏 | `%` | 垂直分割，焦点留在左方 |
| 切换窗格 | `o` / `↑↓←→` | 循环切换或定向跳转 |
| 关闭窗格 | `x` | 需确认（或直接在窗格内输 `exit` / `Ctrl+d`） |
| 调整窗格大小 | `Ctrl+↑↓←→` | 持续按住可连续调整 |
| 显示窗格编号 | `q` | 快速定位 |
| 交换窗格位置 | `{` / `}` | 与前/后窗格互换 |

### 8.3 滚动与复制模式

| 操作 | 按键 |
|------|------|
| 进入滚动模式 | `Ctrl+b [` |
| 翻页 / 移动 | `↑↓` / `PageUp/Down` / `Ctrl+f`(下一页) / `Ctrl+b`(上一页) |
| 开始选择文本 | `空格键`（进入高亮选择） |
| 复制选中内容 | `回车键` / `Enter`（自动存入 tmux 剪贴板） |
| 粘贴内容 | `Ctrl+b ]` |
| 退出滚动模式 | `q` 或 `Esc` |

> 提示：如果需要复制到系统剪贴板，可配置 `bind-key -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "xclip -selection clipboard"`。

### 8.4 实用配置片段

在 `~/.tmux.conf` 中添加：

```bash
# 鼠标支持（滚动、点击切换窗格、调整窗格大小）
set -g mouse on

# 状态栏美化
set -g status-style 'bg=#333333 fg=#ffffff'

# 窗口编号从 1 开始
set -g base-index 1
setw -g pane-base-index 1

# 减少 ESC 延迟（Vim 用户必加）
set -sg escape-time 0

# 历史行数
set -g history-limit 10000

# 分屏时保持当前路径
bind '"' split-window -v -c "#{pane_current_path}"
bind '%' split-window -h -c "#{pane_current_path}"

# 用 | 和 - 代替 % 和 " 分屏（更直觉）
bind '|' split-window -h -c "#{pane_current_path}"
bind '-' split-window -v -c "#{pane_current_path}"
```

修改后生效：`tmux source-file ~/.tmux.conf` 或在 tmux 内 `Ctrl+b :` 输入 `source-file ~/.tmux.conf`。

---

## 9. 行业黑话速查

程序员日常沟通中充斥着各种缩写和行话，新人往往一头雾水。以下按场景分类整理，帮助你快速"听懂人话"。

### 9.1 代码审查与协作

| 黑话 | 全称 | 含义 |
|------|------|------|
| **WIP** | Work In Progress | 还没做完，别合（PR 标题常见） |
| **PR** | Pull Request | GitHub 上的合并请求 |
| **MR** | Merge Request | GitLab 上的合并请求，和 PR 本质相同 |
| **CR** | Code Review | 代码审查 |
| **TL;DR** | Too Long; Didn't Read | 太长不看，通常后面跟一段总结 |
| **Nit / Nitpick** | Nitpick | 小问题挑剔，不影响功能的细节建议 |
| **RFC** | Request For Comments | 提案征集意见，常用于架构设计讨论 |
| **CC** | Carbon Copy | 抄送（邮件/PR 中） |


### 9.2 架构与设计原则

| 黑话 | 全称 | 含义 |
|------|------|------|
| **KISS** | Keep It Simple, Stupid | 保持简单，别过度设计 |
| **DRY** | Don't Repeat Yourself | 不要重复自己，抽取复用 |
| **YAGNI** | You Aren't Gonna Need It | 你不会需要它的，别提前过度设计 |
| **SOLID** | 五大面向对象原则 | SRP/OCP/LSP/ISP/DIP 的缩写 |
| **POC** | Proof Of Concept | 概念验证，先跑通再说 |
| **MVP** | Minimum Viable Product | 最小可行产品 |
| **IaC** | Infrastructure as Code | 基础设施即代码（Terraform、Ansible） |
| **DDD** | Domain-Driven Design | 领域驱动设计 |
| **BFF** | Backend For Frontend | 为前端服务的后端中间层 |
| **SPOF** | Single Point Of Failure | 单点故障 |
| **DSL** | Domain-Specific Language | 领域特定语言 |

### 9.3 运维与稳定性

| 黑话 | 全称 | 含义 |
|------|------|------|
| **SLA** | Service Level Agreement | 服务等级协议（对客户的承诺） |
| **SLO** | Service Level Objective | 服务等级目标（内部目标） |
| **SLI** | Service Level Indicator | 服务等级指标（衡量 SLO 的数据） |
| **SEV** | Severity | 事故等级（SEV1 最严重） |
| **P0/P1/P2** | Priority 0/1/2 | 优先级，P0 最紧急 |
| **On-call** | On-call | 值班，负责响应线上问题 |
| **Runbook** | Runbook | 运维操作手册，出事照着做 |
| **Toil** | Toil | 重复性、无成长的运维琐事 |
| **Blameless Post-mortem** | — | 无指责复盘，只找根因不追责 |
| **MTTR** | Mean Time To Recovery | 平均恢复时间 |
| **MTBF** | Mean Time Between Failures | 平均故障间隔 |
| **OOM** | Out Of Memory | 内存溢出 |
| **CPU throttle** | — | CPU 被限流（容器/K8s 场景常见） |
| **Page / Alert** | — | 报警通知（源自 PagerDuty） |

### 9.4 产品与数据指标

| 黑话 | 全称 | 含义 |
|------|------|------|
| **DAU** | Daily Active Users | 日活跃用户数 |
| **MAU** | Monthly Active Users | 月活跃用户数 |
| **QPS** | Queries Per Second | 每秒查询数 |
| **TPS** | Transactions Per Second | 每秒事务数 |
| **RPS** | Requests Per Second | 每秒请求数 |
| **PV / UV** | Page View / Unique Visitor | 页面浏览量 / 独立访客数 |
| **GMV** | Gross Merchandise Volume | 成交总额（电商常用） |
| **ROI** | Return On Investment | 投资回报率 |
| **ARPU** | Average Revenue Per User | 每用户平均收入 |
| **LTV** | Life Time Value | 用户生命周期价值 |
| **CAC** | Customer Acquisition Cost | 获客成本 |
| **Churn Rate** | — | 流失率 |
| **Retention Rate** | — | 留存率 |

### 9.5 开发文化与调试行话

| 黑话 | 含义 |
|------|------|
| **Dogfooding** | 自己用自己开发的产品（吃自己的狗粮） |
| **Rubber Ducking** | 橡皮鸭调试法：对着玩具鸭子讲代码逻辑，讲着讲着就找到 bug 了 |
| **Yak Shaving** | 牦牛剃毛：为了解决 A，必须先解决 B，B 又依赖 C……无限嵌套 |
| **Bikeshedding** | 自行车棚效应：在琐事上争论不休，大事反而无人关注 |
| **Tech Debt** | 技术债：为短期速度牺牲的代码质量，迟早要还 |
| **Gold Plating** | 镀金：过度完善本不需要的功能 |
| **Happy Path** | 正常流程（没考虑异常的那条路） |
| **Smoke Test** | 冒烟测试：最基本的功能验证，跑不过就别往下测了 |
| **Flaky Test** | 时过时不过的测试，最烦人 |
| **Regression** | 回归：新改动导致原来正常的功能挂了 |
| **Heisenbug** | 海森堡 bug：一调试就消失，不调试就出现 |
| **Bohrbug** | 玻尔 bug：可稳定复现的 bug（和 Heisenbug 对应） |
| **Moth** | 世上第一个 bug：1947 年一只飞蛾卡在继电器里 |
| **Shotgun Debugging** | 散弹枪调试：瞎改一通看哪个能修 |
| **voodoo programming** | 巫毒编程：复制粘贴一段代码不知道原理但能用 |

### 9.6 安全相关缩写

| 黑话 | 全称 | 含义 |
|------|------|------|
| **XSS** | Cross-Site Scripting | 跨站脚本攻击 |
| **CSRF** | Cross-Site Request Forgery | 跨站请求伪造 |
| **SQLi** | SQL Injection | SQL 注入 |
| **RCE** | Remote Code Execution | 远程代码执行（最严重的漏洞类型之一） |
| **MITM** | Man-In-The-Middle | 中间人攻击 |
| **DoS / DDoS** | (Distributed) Denial of Service | （分布式）拒绝服务攻击 |
| **OWASP** | Open Web Application Security Project | Web 安全标准组织 |
| **CVE** | Common Vulnerabilities and Exposures | 公开漏洞编号 |
| **0-day** | Zero-day | 还没被公开/修补的漏洞 |
| **PE** | Privilege Escalation | 提权 |
| **RBAC** | Role-Based Access Control | 基于角色的访问控制 |

### 9.7 通用缩写与网络用语

| 黑话 | 全称 | 含义 |
|------|------|------|
| **AFAIK** | As Far As I Know | 据我所知 |
| **IMO / IMHO** | In My (Humble) Opinion | 依我（谦逊）之见 |
| **TIL** | Today I Learned | 今天学到了 |
| **RTFM** | Read The F***ing Manual | 自己看文档去（语气不善） |
| **GIYF** | Google Is Your Friend | 自己搜去 |
| **IANAL** | I Am Not A Lawyer | 我不是律师（免责声明） |
| **BOFH** | Bastard Operator From Hell | 暴躁运维（网络梗） |
| **JFGI** | Just F***ing Google It | 自己搜 |
| **FWIW** | For What It's Worth | 不管有没有用，说一下 |
| **IIRC** | If I Recall/Remember Correctly | 如果没记错的话 |
| **AFK** | Away From Keyboard | 离开键盘（不在工位） |

---

## 10. 优质在线资源推荐

### Git

- [Conventional Commits 官方规范](https://www.conventionalcommits.org/zh-hans/v1.0.0/) — 提交规范权威文档（中文）
- [Pro Git 中文版](https://git-scm.com/book/zh/v2) — Git 圣经，免费开源
- [Learn Git Branching](https://learngitbranching.js.org/?locale=zh_CN) — 交互式可视化练习（强推新人）
- [Oh Shit, Git!?!](https://ohshitgit.com/zh) — 救命场景速查
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow) — GitHub 推荐工作流
- [Git Flow 中文](https://nvie.com/posts/a-successful-git-branching-model/) — 经典分支模型

### Linux / Shell

- [linuxcool 命令大全](https://www.linuxcool.com/) — 中文 Linux 命令手册，覆盖 550+ 命令
- [man7.org](https://man7.org/linux/man-pages/) — 官方 man 手册在线版
- [explainshell.com](https://explainshell.com/) — 把 shell 命令拆解解释，神器
- [tldr pages](https://tldr.inbrowser.app/) — 极简命令示例，直接看用法
- [ShellCheck](https://www.shellcheck.net/) — 在线 shell 脚本静态检查
- [Bash 脚本教程（阮一峰）](https://wangdoc.com/bash/) — 中文 bash 完整教程

### 综合速查站

- [devhints.io](https://devhints.io/) — 一站式 cheatsheet，覆盖几百种工具
- [quickref.me](https://quickref.me/) — 高质量速查卡片
- [OverAPI](https://overapi.com/) — 各种语言/工具速查图

### Vim

- [Vim Adventures](https://vim-adventures.com/) — 游戏化学 Vim
- [vimtutor](https://www.openvim.com/) — 浏览器版 vimtutor
- [vim-cheatsheet](https://vim.rtorr.com/lang/zh_cn) — Vim 速查表中文版

### 正则

- [regex101.com](https://regex101.com) — 在线测试 + 解释（首选）
- [regexr.com](https://regexr.com/) — 可视化正则学习
- [正则表达式 30 分钟入门](https://deerchao.cn/tutorials/regex/regex.htm) — 中文经典入门

### Docker

- [Docker 官方文档](https://docs.docker.com/) — 权威
- [Docker — 从入门到实践](https://yeasy.gitbook.io/docker_practice/) — 中文开源书
- [Play with Docker](https://labs.play-with-docker.com/) — 浏览器在线沙盒

### tmux

- [tmux 官方仓库](https://github.com/tmux/tmux) — 源码与文档
- [tmux 入门教程（阮一峰）](https://wangdoc.com/bash/tmux.html) — 中文友好入门
- [tmuxcheatsheet.com](https://tmuxcheatsheet.com/) — 交互式速查表
- [A Quick and Easy Guide to tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) — 英文图文教程

### 其他高频

- [DevDocs](https://devdocs.io/) — 把所有语言/框架文档聚合到一个站
- [Can I use](https://caniuse.com/) — 前端 API 兼容性查询
- [HTTP 状态码速查](https://httpstatuses.com/) — 各种 HTTP 状态码含义
- [crontab.guru](https://crontab.guru/) — cron 表达式实时解析

---

> 持续完善中。如发现错误或想补充，欢迎修改。
