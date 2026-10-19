# 探索工作台

浏览地址：[perpetual-learning GitHub Pages](https://jackie2049.github.io/perpetual-learning/)。

这个 demo 把仓库积累呈现为总览、成果库、工作日志、待决事项和收藏。支持标题/主题搜索、分类筛选、精选内容站内阅读、最新日志回看；历史材料跳转到 GitHub。收藏只保存在当前浏览器。

首页四个统计面板均可点击：知识笔记与开源材料进入对应分类，工作日志与待决事项进入对应页面。阅读面板支持 `$...$` 行内公式与 `$$...$$` 独立公式；KaTeX 脚本和字体随网页提供，不依赖外部 CDN。代码块保留原样。

## 内容来源

- `build.py` 只读取当前 Git 提交里的 Markdown；未跟踪与未提交文件不会发布。历史资料仅索引标题、章节和路径，正文保留在 GitHub。
- `featured.json` 选择首页的六份成果并维护简短导读；导读是人工整理的阅读入口。只有这些成果、最新日期日志及已选待决文档输出站内阅读页面。
- 最新日志的“当日回看”和时间条目直接从原文提取。各文档链接固定到构建使用的提交，页面显示来源快照，不报告实时 Agent 状态。
- demo 的待决文档清单在 `build.py` 中，状态摘自各文档的状态行；已发布记录自动退出该清单。新增待决方向需要维护清单，页面不替代完整 WORK。发布决定与运行控制仍在 Agent 会话中处理。
- 内容提交并推送至 main 后，运行发布命令重建网页快照。精选导读与待决清单随课题变化维护，网页不自动推断新课题。

## 本地运行

Python 3.9+，建议使用独立虚拟环境：

```bash
python3 -m venv /tmp/perpetual-learning-web
/tmp/perpetual-learning-web/bin/pip install -r web/requirements.txt
/tmp/perpetual-learning-web/bin/python -m unittest discover -s web -p 'test_*.py'
/tmp/perpetual-learning-web/bin/python web/build.py
python3 -m http.server 8765 --directory web/dist --bind 127.0.0.1
```

打开 `http://127.0.0.1:8765`。网页文件改动后重新构建即可预览；内容文档需要提交后才进入快照。`web/dist/` 为忽略的构建输出。

## 发布

使用 [publish.py](publish.py) 将生成结果发布到独立的 `gh-pages` 分支；main 仅维护源码。仓库 Pages 来源设为该分支根目录，GitHub 自带 Pages 部署负责上线。

```bash
# 先提交并推送内容和网页源码，再使用已安装构建依赖的 Python：
/tmp/perpetual-learning-web/bin/python web/publish.py
```

发布器核对 HEAD 与远端 main 一致，在临时目录生成发布提交，不切换当前工作分支，不强推。已有 gh-pages 分支必须带本工具的生成标记才允许更新。没有配置定时同步；更新网页需运行上述命令。


页面使用相对资源路径和 hash 导航，适用于仓库子路径。渲染 Markdown 时移除脚本和不安全链接；历史资料的验证范围和外部状态按原记录理解。

GitHub Pages 分支发布参考：[GitHub 官方文档](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。
