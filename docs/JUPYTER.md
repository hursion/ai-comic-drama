# JupyterLab 操作、端口映射、保活与断点续跑

## 1. 进入实例

1. 魔搭 → 我的 Notebook → 选择带 **AMD / 192G** 标记的实例 → 启动。
2. 等待状态变成“运行中”，点“打开 JupyterLab”。
3. 首次进入会提示：只有 `/mnt/workspace` 下的文件会持久化。**不要把模型放到 `/root` 或 `/tmp`，除非你接受重启后丢失。**
4. 右上角可以看到剩余额度；开始/结束各记一次。

## 2. 上传项目

方式一（推荐）：把本仓库打成 zip，在 JupyterLab 左侧文件树上传到 `/mnt/workspace/`，然后：

```bash
cd /mnt/workspace
unzip ai-comic-drama.zip
mv ai-comic-drama-main ai-comic-drama   # 如果解压目录名不同
cd ai-comic-drama
```

方式二：如果实例能访问你的 Git 仓库：

```bash
cd /mnt/workspace
git clone <your-repo> ai-comic-drama
```

方式三：在 Jupyter 里新建 notebook，用 `%%writefile` 逐个写文件（不推荐，容易漏）。

## 3. 打开哪个 notebook

| Notebook | 用途 |
|---|---|
| `notebooks/00_setup.ipynb` | 预检 → 安装 → 下载 Set A → 启动 ComfyUI |
| `notebooks/01_episode.ipynb` | 分镜 → 关键帧 → 视频 → 配音 → 合成 |

Notebook 里的命令都以 `!bash` / `!python` 调用仓库脚本，逻辑不在 notebook 里，方便版本管理。

## 4. 端口映射（在浏览器看 ComfyUI）

批处理不需要端口映射；只有想在浏览器里拖工作流时才需要。

1. 启动 ComfyUI：`bash scripts/03_start_comfyui.sh`
2. 确认本机可访问：`curl -s http://127.0.0.1:8188/system_stats`
3. 回到 Notebook 页面 → 右上角“端口映射” → 新增 `8188` → 保存。
4. 用平台生成的地址访问。常见形式：
   - `https://<实例域名>/proxy/8188/`
   - 或平台分配的独立 URL
5. 如果页面 404 / 502：
   - 确认 ComfyUI 监听的是 `0.0.0.0` 而不是 `127.0.0.1`；
   - 确认端口映射已保存并生效；
   - 看 `logs/comfyui.log` 是否有启动失败；
   - 换一个端口（8190）再试。

> 不要把无鉴权的 ComfyUI 直接暴露到公网长期运行；调试完关掉端口映射。

## 5. 会话保活

平台通常在 **1 小时无操作**后自动关机。最容易踩坑的是“下载完了，人去吃饭，回来实例没了”。

做法：

1. 保持 JupyterLab 标签页打开；
2. 在 notebook 里跑一个保活 cell：

```python
!nohup python scripts/04_keepalive.py --interval 60 --log logs/keepalive.log > /dev/null 2>&1 &
```

3. `scripts/04_keepalive.py` 会每 60 秒访问一次 ComfyUI `/system_stats`，并写一行心跳日志：

```bash
tail -f logs/keepalive.log
```

4. 保活不是万能的：平台可能按“浏览器活跃度”而不是“进程活跃度”判断。**离开超过 10 分钟，最稳的做法是手动关闭实例**，回来再启动；持久盘里的模型不会丢。

## 6. 长任务的正确姿势

- **拆小**：一次批处理不超过 20–30 分钟；每批结束把输出写到持久盘。
- **可续跑**：`run_batch` 默认跳过已存在的输出文件；中断后重跑同一条命令即可。
- **日志落盘**：`nohup ... > logs/batch_$(date +%H%M).log 2>&1 &`。
- **先测速**：先跑 3 个镜头，记录平均耗时，再决定是否全量。
- **检查磁盘**：每批前后 `df -h /mnt/workspace`；低于 10GB 就清理。

## 7. 断点续跑

### 7.1 实例重启后

```bash
cd /mnt/workspace/ai-comic-drama
source venv/bin/activate
bash scripts/03_start_comfyui.sh
curl -s http://127.0.0.1:8188/system_stats
```

模型、代码、输出都在持久盘，不需要重新下载；如果 ComfyUI 或自定义节点有更新，先跑 G1/G3。

### 7.2 批处理中断

```bash
# 已存在的 keyframes/clips 会被跳过
python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage images
python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage videos
```

### 7.3 某个镜头反复失败

1. 在 `episode.json` 里给该镜头加 `"enabled": false`；
2. 重新生成其余镜头；
3. 单独修这个镜头（换参考图/换构图/换 Seed），修好后改回 `true`；
4. 最后合成时跳过 disabled 镜头，或用占位镜头补时长。

## 8. Jupyter 常用技巧

```bash
# 看 GPU
!rocm-smi

# 看显存占用（ComfyUI 进程）
!rocm-smi --showmeminfo vram

# 看磁盘
!df -h /mnt/workspace

# 看 ComfyUI 日志
!tail -n 100 logs/comfyui.log

# 杀 ComfyUI
!kill $(cat logs/comfyui.pid) || true

# 看端口
!ss -lntp | grep 8188
```

在 notebook 中执行 shell 命令时，工作目录默认是 notebook 所在目录；建议每个 cell 都用绝对路径，或先 `%cd /mnt/workspace/ai-comic-drama`。

## 9. 常见问题

| 问题 | 处理 |
|---|---|
| Jupyter kernel 死了 | 重启 kernel；ComfyUI 是独立进程，不受影响 |
| 浏览器断开后 ComfyUI 还在跑 | 正常；用 `curl` 检查，重连后继续 |
| 端口映射地址打不开 | 换端口；确认 `--listen 0.0.0.0`；看日志 |
| 上传大文件很慢 | 先压缩；或从 ModelScope 直接下载到实例 |
| 实例启动失败 | 换时段/重试；查看平台公告；不要反复狂点 |
| 磁盘满导致 Jupyter 异常 | 删缓存/旧输出；`du -sh * | sort -h` 定位 |
| 忘记关实例，额度被烧 | 设手机提醒；每次收工按 CHECKLIST 执行 |

## 10. 安全

- 不要在 notebook 里硬编码 ModelScope/HF token；用环境变量或 `~/.cache` 配置。
- 端口映射用完即关。
- 不要把未公开的剧本、角色设定、客户素材放到公开仓库。
- 实例是共享云环境，敏感数据落盘前确认平台的数据政策。
