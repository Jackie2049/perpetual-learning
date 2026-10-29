# JackiesGallery 设计规范

> 本站展品设计的单一事实源。新增展品前先读这里。

## 原则

1. **自包含** — 每份展品是单个 HTML 文件，CSS/JS 全部内联，零外部依赖、零外链字体/库/CDN。离线双击可看，分享即所见。
2. **主题适配** — 支持浅/深色：浅色为默认，深色用 `@media (prefers-color-scheme: dark)` 覆盖。颜色通过 `:root` 的 CSS 变量定义，暗色只改变量，不改结构。
3. **克制** — 统一色系、低饱和；导航/入口页浅色为主，内容页可深色。不堆装饰。

## 目录与登记

```
<作品名>/
└── index.html          # 展品本体（自包含）
```

顶层 `index.html` 的 `EXPERIMENTS` 数组登记：
```js
const EXPERIMENTS = [
  { title: '展品名', desc: '一句话描述', url: '<作品名>/', date: 'YYYY-MM' }
];
```

## 发布

推送 `main` 分支 → GitHub Pages 自动重建 → `https://jackie2049.github.io/JackiesGallery/<作品名>/` 生效。
