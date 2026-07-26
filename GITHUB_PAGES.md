# 如何用 GitHub Pages 分享本报告

本目录已包含静态页 `index.html`（自动引用 `output_*/figures/` 图片）。

## 1. 把文件推到 `联调成果` 分支

确保分支根目录有：

- `index.html`
- `实验报告_Mock与DeepSeek对比.md`
- `output_mock_100x50/`（含 figures）
- `output_ds_100x20/`（含 figures）
- 其它 config / README 等

## 2. 开启 GitHub Pages

1. 打开仓库 **Settings → Pages**
2. **Source** 选 **Deploy from a branch**
3. Branch 选 **`联调成果`**，文件夹选 **`/ (root)`**
4. 点 **Save**，等 1–2 分钟

## 3. 访问链接

一般形如：

```text
https://jiaxinchen-annie.github.io/SURF-0245-Multi-Agent-Social-Simulation-for-Public-Opinion-Governance/
```

若仓库名或用户名大小写不同，以 Settings → Pages 页面显示的 URL 为准。

## 4. 本地预览（不上传也能看）

双击 `index.html`，或在目录下执行：

```bat
cd /d "E:\SURF\SURF C\output_上传包"
start index.html
```

图片路径为相对路径，需与 `output_*` 文件夹在同一层级。
