import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
from collections import Counter

# ==============================================================================
#  组件一： CNN 预测器 
# ==============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # 输入: (B, 1, 128, 128)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (B, 16, 64, 64)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (B, 32, 32, 32)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (B, 64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> (B, 128, 8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), # -> (B, 128 * 8 * 8)
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes) # 输出层
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class LetterPredictor:
    def __init__(self, model_path='wordle_recognizer_torch.pth', class_names_path='class_names.txt', target_size=128):
        """
        初始化预测器。

        Args:
            model_path (str): 训练好的模型权重文件 (.pth) 路径。
            class_names_path (str): 包含类别名称的文本文件路径。
            target_size (int): 模型输入的图像尺寸。
        """
        self.target_size = target_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载类别名称
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            num_classes = len(self.class_names)
        except Exception as e:
            print(f"错误: 无法加载类别文件 '{class_names_path}': {e}")
            self.model = None
            return

        # 2. 实例化模型并加载权重
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        try:
            # 加载状态字典
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            # 切换到评估模式 (非常重要！这会禁用Dropout等)
            self.model.eval()
            print(f"模型 '{model_path}' 加载成功，运行在 {self.device}。")
        except Exception as e:
            print(f"错误: 无法加载模型 '{model_path}': {e}")
            self.model = None

        # 3. 定义图像预处理转换
        self.transform = transforms.Compose([
            # transforms.ToPILImage(), # 确保输入是PIL Image
            # 将任意尺寸的输入图，先填充黑边再缩放/裁剪到目标尺寸
            transforms.Lambda(lambda img: self.pad_and_resize(img)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def pad_and_resize(self, img):
        """
        一个健壮的函数，将任意输入图像填充为正方形，然后缩放到目标尺寸。
        """
        w, h = img.size
        # 计算填充量，使其成为正方形
        if w > h:
            padding = (0, (w - h) // 2)
        else:
            padding = ((h - w) // 2, 0)
        
        # 使用transforms来填充
        pad_transform = transforms.Pad(padding, fill=0, padding_mode='constant')
        resized_transform = transforms.Resize((self.target_size, self.target_size))
        
        return resized_transform(pad_transform(img))


    def predict(self, image_input):
        """
        对单个字母图片进行预测。

        Args:
            image_input (str or np.ndarray): 图片的文件路径，或者一个OpenCV/Numpy格式的图像数组。

        Returns:
            tuple: (预测的字母, 置信度) 或者 (None, 0) 如果预测失败。
        """
        if self.model is None:
            return None, 0.0

        try:
            # 如果输入是文件路径，用PIL加载
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    print(f"错误: 路径不存在 '{image_input}'")
                    return None, 0.0
                img = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    image_input = image_input[:, :, ::-1] # BGR to RGB
                img = Image.fromarray(image_input)
            else:
                print("错误: 输入必须是文件路径或Numpy数组。")
                return None, 0.0

            # 应用预处理
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # 进行预测
            with torch.no_grad():
                outputs = self.model(img_tensor)
                # 使用softmax获取概率分布
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # 找到最高概率的类别
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                predicted_letter = self.class_names[predicted_idx.item()]
                confidence_percent = confidence.item() * 100
                
                return predicted_letter, confidence_percent

        except Exception as e:
            print(f"预测过程中发生错误: {e}")
            return None, 0.0


# ==============================================================================
#  组件二： 图像分析 (分割、颜色识别)
# ==============================================================================

# 定义游戏中各种颜色的 BGR 值
COLOR_GREEN = np.array([100, 170, 106])
COLOR_YELLOW = np.array([88, 180, 201])
COLOR_GRAY = np.array([126, 124, 120])

def analyze_game_board(image_path, predictor, grid_params, debug_mode=False):
    """
    完整分析游戏截图，返回游戏状态。
    在调试模式下，会打印详细的颜色计算日志，并保存一张带标注的可视化图片。
    """
    
    # --- 内部辅助函数，用于颜色判断 ---
    def get_color_details(block):
        # 使用左上角一个10x10小块的平均颜色来判断，以提高鲁棒性
        corner_patch = block[5:15, 5:15]
        avg_color = np.mean(corner_patch, axis=(0, 1))
        
        distances = {
            "green": np.linalg.norm(avg_color - COLOR_GREEN),
            "yellow": np.linalg.norm(avg_color - COLOR_YELLOW),
            "gray": np.linalg.norm(avg_color - COLOR_GRAY)
        }
        
        final_decision = min(distances, key=distances.get)
        return final_decision, avg_color, distances

    # --- 主函数逻辑开始 ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 错误: 无法加载图片 '{image_path}'")
        return None, 0

    # 仅在调试模式下创建可视化图像的副本
    vis_image = image.copy() if debug_mode else None

    img_h, img_w, _ = image.shape
    game_state = []
    detected_word_length = 0
     
    gp = grid_params
    total_grid_w = img_w - gp['margin_left'] - gp['margin_right'] - (gp['gap_x'] * (gp['num_cols'] - 1))
    block_w = total_grid_w // gp['num_cols']
    total_grid_h = img_h - gp['margin_top'] - gp['margin_bottom'] - (gp['gap_y'] * (gp['num_rows'] - 1))
    block_h = total_grid_h // gp['num_rows']

    print("\n--- 棋盘分析中 ---")
    if debug_mode:
        print("--- 开启详细调试模式 ---")

    for r in range(gp['num_rows']):
        row_letters = []
        for c in range(gp['num_cols']):
            start_x = gp['margin_left'] + c * (block_w + gp['gap_x'])
            start_y = gp['margin_top'] + r * (block_h + gp['gap_y'])
            end_x = start_x + block_w
            end_y = start_y + block_h
            
            block_original = image[start_y : end_y, start_x : end_x]

            # 跳过空方块
            if np.mean(block_original) < 30 or np.mean(block_original) > 240:
                # 在调试模式下，依然画出空方块的分割框
                if debug_mode:
                    cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), (200, 200, 200), 1)
                continue

            # --- 颜色和字母识别 ---
            color, avg_color, color_distances = get_color_details(block_original)
            
            lower_white = np.array([200, 200, 200])
            upper_white = np.array([255, 255, 255])
            mask = cv2.inRange(block_original, lower_white, upper_white)
            black_background = np.zeros_like(block_original)
            binarized_block = cv2.bitwise_or(black_background, black_background, mask=mask)
            binarized_block[mask != 0] = [255, 255, 255]
            letter, conf = predictor.predict(binarized_block)
            
            # --- 信息处理与调试输出 ---
            if letter:
                game_state.append({'letter': letter, 'color': color, 'position': c})
                color_map = {"green": "Gr", "yellow": "Yl", "gray": "Gy"}
                row_letters.append(f" {letter}({color_map.get(color, '?')}) ")

                if debug_mode:
                    # 打印颜色计算的详细日志
                    print(f"\n[DEBUG] 方块 R{r+1}, C{c+1}:")
                    avg_color_int = [int(x) for x in avg_color]
                    print(f"  - 计算出的平均颜色 (BGR): {avg_color_int}")
                    print("  - 到各标准色的距离:")
                    for color_name, dist in color_distances.items():
                        print(f"    - 距离 {color_name.capitalize():<7}: {dist:.2f}")
                    print(f"  - ===> 颜色判断: {color.upper()}")
                    print(f"  - ===> 字母判断: {letter} (置信度: {conf:.1f}%)")

                    # 在可视化图像上绘制所有信息
                    cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
                    label_text = f"{letter} ({conf:.1f}%)"
                    text_color = (0, 0, 255)
                    text_pos = (start_x + 5, start_y + 20)
                    cv2.putText(vis_image, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    color_label_pos = (start_x + 5, end_y - 10)
                    cv2.putText(vis_image, color, color_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
        if row_letters:
            if detected_word_length == 0:
                detected_word_length = len(row_letters)
            print(f"第 {r+1} 行 (长度 {len(row_letters)}): {''.join(row_letters)}")
            
    # --- 收尾 ---
    if not game_state:
        print("未在图片中检测到任何已猜测的单词。")
    print("--- 分析完成 ---\n")

    # 仅在调试模式下保存最终的可视化图像
    if debug_mode:
        output_vis_path = "debug_visualization.png"
        cv2.imwrite(output_vis_path, vis_image)
        print(f"✅ 已将调试可视化图像保存到 '{output_vis_path}'")
    
    return game_state, detected_word_length

# ==============================================================================
#  组件三： Wordle 求解逻辑
# ==============================================================================

def filter_word_list(words, game_state, debug_mode=False, debug_word=None):
    """ 
    根据游戏状态筛选单词列表。
    修正了规则整理逻辑，绿色优先。
    """
    if debug_mode and debug_word:
        debug_word = debug_word.upper()
        print(f"\n--- 调试筛选过程: 追踪单词 '{debug_word}' ---")

    # 1. 整理规则 (greens, yellows, grays) 
    greens = {}   # {position: letter}
    yellows = {}  # {letter: [list_of_banned_positions]}
    grays = set()

    # --- 步骤 1a: 优先处理所有绿色字母 ---
    for info in game_state:
        if info['color'] == 'green':
            greens[info['position']] = info['letter']
    
    # 获取所有已被确认为绿色的字母集合
    green_letters = set(greens.values())

    # --- 步骤 1b: 处理黄色和灰色，并忽略已变绿的字母 ---
    for info in game_state:
        letter, color, pos = info['letter'], info['color'], info['position']
        
        # 如果这个字母已经是绿色了，就不要再处理它作为黄色或灰色的情况
        if letter in green_letters:
            continue
            
        if color == 'yellow':
            if letter not in yellows: yellows[letter] = []
            yellows[letter].append(pos)
        elif color == 'gray':
            # 同样，如果一个字母在别处是黄色，也不应是灰色
            is_yellow_elsewhere = any(letter == y_letter for y_letter in yellows.keys())
            if not is_yellow_elsewhere:
                grays.add(letter)

    if debug_mode and debug_word:
        print("【规则整理结果】:")
        print(f"  - 绿色规则 (greens): {greens}")
        print(f"  - 黄色规则 (yellows): {yellows}")
        print(f"  - 灰色规则 (grays): {grays}")
        print("-----------------------------------------")

    # --- 步骤 1c: 修正字母数量的计算逻辑 ---
    # 数量只由绿色和黄色字母决定，且每个字母只计一次最高状态
    letter_counts = Counter(greens.values())
    letter_counts.update(yellows.keys())

    possible_words = []
    for word in words:
        word = word.upper()
        is_debug_target = (debug_mode and word == debug_word)
        
        if is_debug_target: print(f"\n【开始检查单词 '{word}'】")
        valid = True

        # [1] 绿色检查
        for pos, letter in greens.items():
            if word[pos] != letter: valid = False; break
        if not valid: continue

        # [2] 黄色检查
        for letter, banned_positions in yellows.items():
            if letter not in word: valid = False; break
            for pos in banned_positions:
                if word[pos] == letter: valid = False; break
            if not valid: break
        if not valid: continue

        # [3] 灰色检查
        for letter in grays:
            if letter in word: valid = False; break
        if not valid: continue


        # --- 规则4: 字母数量检查 ---
        if is_debug_target: print("  [4] 检查字母数量规则...")
        # 检查单词中每个相关字母的数量是否至少等于我们已知的数量
        for letter, required_count in letter_counts.items():
            if word.count(letter) < required_count:
                if is_debug_target:
                    print(f"    ❌ 失败: 单词中字母 '{letter}' 的数量 ({word.count(letter)}) 少于线索中要求的数量 ({required_count})。")
                valid = False; break
        if not valid: continue
        if is_debug_target: print("    ✅ 通过")

        possible_words.append(word)
        if is_debug_target: print(f"🎉 '{word}' 通过所有筛选，已加入可能列表。")

    return possible_words

def suggest_best_word(possible_words, all_words):
    """ 从可能的单词中，推荐一个最佳猜测 """
    if not possible_words:
        return "🤔 根据当前线索在词库中找不到任何可能的单词。", []
    if len(possible_words) <= 2:
        return f"答案很可能是: {possible_words[0]}", possible_words

    # 策略：从所有单词（不仅仅是可能的答案）中，找一个能最大化排除可能性的词
    letter_freq = Counter()
    for word in possible_words:
        letter_freq.update(set(word))

    best_score = -1
    best_word = ""
    
    # 只从可能答案中选择
    word_pool = all_words if len(possible_words) > 10 else possible_words
    
    for word in word_pool:
        # 使用 set(word) 确保每个字母只计分一次
        score = sum(letter_freq[letter] for letter in set(word))
        if score > best_score:
            best_score = score
            best_word = word
            
    return f"最佳猜测是: {best_word} (能提供最多信息)", possible_words


def main_solver_logic(wordlist_path, image_path):
    """
    主逻辑函数，它执行所有计算并返回结构化数据。
    所有的 print() 语句都会被 app.py 捕获为日志
    """
    # 1. 初始化和加载资源
    GRID_PARAMS = {
        'num_rows': 6, 'num_cols': 5, 'margin_top': 10, 'margin_bottom': 21,
        'margin_left': 16, 'margin_right': 16, 'gap_x': 6, 'gap_y': 6
    }
    
    try:
        with open(wordlist_path, 'r') as f:
            full_word_library = [line.strip().upper() for line in f]
    except FileNotFoundError:
        print(f"❌ 错误: '{wordlist_path}' 未找到。请确保词库文件存在。")
        return "词库文件未找到", []

    # predictor 的初始化会打印加载信息，这会被捕获为日志
    predictor = LetterPredictor()
    if not predictor.model:
        print("❌ 模型加载失败，程序退出。")
        return "CNN模型加载失败", []

    # 2. 分析图像获取游戏状态
    # analyze_game_board 内部的 print 也会被捕获
    game_state, word_length = analyze_game_board(image_path, predictor, GRID_PARAMS)

    if not game_state:
        # 如果没有识别到任何东西，返回一个建议
        return "💡 未检测到棋盘信息。建议的起始词 (5字母): RAISE 或 **SOARE", []

    print(f"✅ 已检测到单词长度为: {word_length}")

    # 3. 根据检测到的长度筛选词库
    all_words = [w for w in full_word_library if len(w) == word_length]
    if not all_words:
        print(f"❌ 错误: 在词库中没有找到任何长度为 {word_length} 的单词。")
        return f"词库中缺少长度为 {word_length} 的单词", []
        
    print(f"已从词库中加载 {len(all_words)} 个长度为 {word_length} 的单词。")

    # 4. 求解
    possible_words = filter_word_list(all_words, game_state)
    
    # 5. 获取建议和可能性列表
    # suggest_best_word 函数返回 (建议字符串, 可能性列表)
    suggestion, possibilities = suggest_best_word(possible_words, all_words)

    # 6. 返回结构化数据，供 app.py 使用
    return suggestion, possibilities