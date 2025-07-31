# Wordle辅助工具

## 项目简介

本项目是一个基于Flask的Web应用，结合了图像识别与Wordle求解，能够自动识别Wordle游戏截图，分析当前局面，并给出猜测建议。只需上传Wordle游戏截图，会自动识别字母及颜色状态，筛选可能答案，并推荐下一步操作。

## 主要功能

- 上传Wordle游戏截图，自动识别棋盘字母及颜色（绿/黄/灰）。
- 分析当前局面，筛选所有可能答案。
- 推荐最佳猜测词。
- 支持深色/浅色主题切换。

## 运行方式

1. **环境准备**
   - Python 3.8+
   - 推荐使用虚拟环境

2. **安装依赖**

   ```sh
   pip install -r requirements.txt
   ```

3. **准备模型和词库**

   - 确保根目录下有 `wordle_recognizer_torch.pth`（CNN模型权重）、`class_names.txt`（单词列表）和 `wordlist.txt`（单词库）。

4. **启动服务**

   ```sh
   python app.py // 方式一
   flask run  //方式二
   ```

   启动后访问 [http://localhost:5000](http://localhost:5000) 即可使用。

## 目录结构

```
.
├── app.py                  # Flask主程序
├── solver_logic.py         # 图像识别与Wordle求解核心逻辑
├── requirements.txt        # 依赖包列表
├── wordle_recognizer_torch.pth  # 字母识别CNN模型权重
├── wordlist.txt            # Wordle单词库
├── static/
│   ├── style.css           # 前端样式
│   └── uploads/            # 上传图片存放目录
└── templates/
    └── index.html          # 前端页面模板
```

## 主要文件说明

- app.py：Web服务入口，处理文件上传与结果展示。
- solver_logic.py：包含CNN字母识别、棋盘分析、Wordle求解等核心算法。
- index.html：主页面模板，支持拖拽上传、主题切换等交互。
- style.css：美观的响应式样式，支持深浅主题。
- wordle_recognizer_torch.pth：PyTorch训练好的字母识别模型。
- wordlist.txt：Wordle可用单词列表。

## 使用说明
- 字母切分写得很死，有需要用到的，自行修改这部分逻辑吧……
- 词库当前仅有5字单词，如需扩展请自行替换`wordlist.txt`
1. 打开网页，上传Wordle游戏截图
2. 系统分析图片，识别字母与颜色，输出最佳猜测及所有可能答案。
3. 查看详细分析日志，辅助理解推荐结果。

