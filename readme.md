# Wordle辅助工具

## 项目简介

本项目是一个基于Flask的Web应用，结合了图像识别与Wordle求解，能够自动识别Wordle游戏截图，分析当前局面，并给出猜测建议。只需上传Wordle游戏截图，会自动识别字母及颜色状态，筛选可能答案，并推荐下一步操作。

## 主要功能

- 上传Wordle游戏截图，自动识别棋盘字母及颜色（绿/黄/灰）。
- 分析当前局面，筛选所有可能答案。
- 推荐最佳猜测词。
- 支持深色/浅色主题切换。

## 运行方式

1. 环境准备
   - Python 3.8+
   - 推荐使用虚拟环境

2. 安装依赖

   ```sh
   pip install -r requirements.txt
   ```

3. 准备模型和词库

   - 确保根目录下有 `wordle_recognizer_torch.pth`（CNN模型权重）、`class_names.txt`（单词列表）和 `wordlists`（单词库）。

4. 启动服务

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
├── wordle_recognizer.onnx  # 字母识别CNN模型权重
├── wordlist                # Wordle单词库
├── static/
│   ├── style.css           # 前端样式
│   └── uploads/            # 上传图片存放目录
└── templates/
    └── index.html          # 前端页面模板
```

## 主要文件说明

- app.py：Web服务入口，处理文件上传与结果展示。
- solver_logic.py：包含CNN字母识别、棋盘分析、Wordle求解等算法
- index.html：主页面模板，支持拖拽上传、主题切换等交互
- style.css：样式，支持深浅主题。
- wordle_recognizer.onnx：训练好的字母识别模型,转换为onnx格式
- wordlists文件夹：Wordle可用单词列表。

## 使用说明
1. 打开网页，上传Wordle游戏截图
2. 系统分析图片，识别字母与颜色，输出最佳猜测及所有可能答案

---

### 更新

*   模型更换：识别模块从`.pth` 迁移至轻量级的 ONNX Runtime (`.onnx`)，减小了运行依赖和应用体积，启动速度变快
*   实现动态网格检测：取代了原有的静态坐标切分方式，现在程序能自动定位不同尺寸和长度的字母网格
*   增强通用性：支持 3-12 个字母长度的单词识别，会根据自动检测到的长度加载相应的词库文件
*   模型优化：优化预处理后重新训练模型，鲁棒性更好
*   依赖变更：
    *   移除：`torch`, `torchvision`
    *   新增：`onnxruntime`