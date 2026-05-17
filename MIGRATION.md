# Jekyll 到 Hugo PaperMod 迁移指南

## 迁移完成 ✅

你的博客已成功从 Jekyll Academic Pages 迁移到 Hugo PaperMod 主题。

## 主要变更

### 1. 静态站点生成器
- **之前**: Jekyll (Ruby)
- **现在**: Hugo (Go)

### 2. 主题
- **之前**: Academic Pages (基于 Minimal Mistakes)
- **现在**: PaperMod

### 3. 目录结构变更

#### 之前的 Jekyll 结构:
```
_posts/          # 博客文章
_pages/          # 页面
_layouts/        # 布局模板
_includes/       # 包含模板
_sass/           # SCSS 样式
_config.yml      # 配置文件
```

#### 现在的 Hugo 结构:
```
content/posts/   # 博客文章
content/         # 页面
themes/          # 主题
static/          # 静态文件
hugo.yaml        # 配置文件
```

### 4. 本地开发命令

#### 之前 (Jekyll):
```bash
bundle install
jekyll serve -l -H localhost
# 访问: http://localhost:4000
```

#### 现在 (Hugo):
```bash
hugo server -D
# 访问: http://localhost:1313
```

### 5. 创建新文章

#### 之前 (Jekyll):
文件名: `_posts/2026-05-17-my-post.md`
```yaml
---
title: 'My Post'
date: 2026-05-17
tags:
  - tag1
---
```

#### 现在 (Hugo):
文件名: `content/posts/2026-05-17-my-post.md`
```yaml
---
title: "My Post"
date: 2026-05-17
draft: false
tags:
  - tag1
ShowToc: true
ShowReadingTime: true
---
```

## 已迁移的内容

✅ 所有博客文章 (7 篇)
- 2012-08-14-blog-post-1.md
- 2013-08-14-blog-post-2.md
- 2014-08-14-blog-post-3.md
- 2015-08-14-blog-post-4.md
- 2024-11-17-reinforcement-learning-intro.md
- 2026-05-17-python-list-tutorial.md
- 2199-01-01-future-post.md

✅ About 页面
✅ 静态资源 (images, files)
✅ GitHub Actions 自动部署配置

## 下一步操作

### 1. 预览你的新博客
```bash
hugo server -D
```
然后打开浏览器访问 http://localhost:1313

### 2. 自定义配置
编辑 `hugo.yaml` 文件来自定义:
- 网站标题和描述
- 社交链接
- 主题设置 (深色/浅色模式)
- 菜单项

### 3. 提交并推送
```bash
git add .
git commit -m "Migrate from Jekyll to Hugo PaperMod"
git push origin main
```

### 4. 配置 GitHub Pages
1. 进入仓库设置
2. 导航到 Pages 部分
3. 确保 Source 设置为 "GitHub Actions"
4. GitHub Actions 会自动构建和部署你的站点

## PaperMod 特性

PaperMod 主题提供以下特性:
- 🚀 快速加载
- 🌓 自动深色/浅色模式切换
- 📱 响应式设计
- 🔍 内置搜索功能
- 📊 阅读时间显示
- 🏷️ 标签和分类
- 📄 目录导航 (TOC)
- 🔗 社交图标链接
- 📝 代码高亮
- 🌐 SEO 优化

## 配置示例

### 启用个人主页模式
在 `hugo.yaml` 中:
```yaml
params:
  profileMode:
    enabled: true
    title: "Your Name"
    subtitle: "Your Bio"
    imageUrl: "/images/profile.png"
    imageWidth: 120
    imageHeight: 120
    buttons:
      - name: Posts
        url: posts
      - name: About
        url: about
```

### 添加社交图标
```yaml
params:
  socialIcons:
    - name: github
      url: "https://github.com/yourusername"
    - name: twitter
      url: "https://twitter.com/yourusername"
    - name: email
      url: "mailto:you@example.com"
```

## 常见问题

### Q: 如何更改主题颜色?
A: 在 `hugo.yaml` 中设置 `defaultTheme: dark` 或 `defaultTheme: light`

### Q: 如何添加新页面?
A: 在 `content/` 目录下创建新的 Markdown 文件

### Q: 如何添加图片?
A: 将图片放在 `static/` 目录，然后在 Markdown 中使用 `/images/filename.jpg` 引用

### Q: 文章中的图片路径需要改吗?
A: 是的，如果之前使用 `/assets/images/` 路径，现在需要改为 `/images/`

## 备份

你的原始 Jekyll 项目已备份为: `SendoRay.github.io.jekyll.backup`

## 参考资源

- [Hugo 官方文档](https://gohugo.io/documentation/)
- [PaperMod 主题文档](https://github.com/adityatelange/hugo-PaperMod/wiki)
- [PaperMod 示例站点](https://adityatelange.github.io/hugo-PaperMod/)
