# chengzhy's Blog

This is my personal blog built with [Hugo](https://gohugo.io/) and [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

## Local Development

### Prerequisites

- Hugo >= 0.146.0 (extended version)
- Go (for Hugo modules)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SendoRay/SendoRay.github.io.git
cd SendoRay.github.io
git submodule update --init --recursive
```

2. Run Hugo server:
```bash
hugo server -D
```

3. Open your browser and navigate to `http://localhost:1313`

## Content Structure

- `content/posts/` - Blog posts
- `content/about.md` - About page
- `static/` - Static files (images, documents, etc.)
- `themes/PaperMod/` - PaperMod theme (git submodule)

## Configuration

Main configuration file: `hugo.yaml`

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the `main` branch via GitHub Actions.

## Writing New Posts

Create a new post in `content/posts/` with the following front matter:

```yaml
---
title: "Your Post Title"
date: 2026-05-17
draft: false
tags:
  - tag1
  - tag2
ShowToc: true
ShowReadingTime: true
---
```

## Migration Note

This site was migrated from Jekyll Academic Pages to Hugo PaperMod in May 2026.
