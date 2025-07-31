import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
import os
import argparse
from collections import Counter

# ==============================================================================
#  组件一： CNN 预测器 
# ==============================================================================
class ONNXPredictor:
    def __init__(self, model_path='wordle_recognizer.onnx', class_names_path='class_names.txt'):
        """
        初始化ONNX预测器，加载ONNX模型和类别名称。
        """
        self.class_names = []
        self.session = None
        self.input_name = None
        self.target_size = 128 # 模型训练时使用的尺寸
        
        print("--- 初始化ONNX预测器 ---")
        # 1. 加载ONNX模型
        try:
            self.session = ort.InferenceSession(model_path)
            # 自动获取模型的输入节点名称
            self.input_name = self.session.get_inputs()[0].name
            print(f"✅ ONNX模型 '{model_path}' 加载成功。")
        except Exception as e:
            print(f"❌ 错误: 无法加载ONNX模型 '{model_path}': {e}")
            return
            
        # 2. 加载类别名称
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✅ 成功加载 {len(self.class_names)} 个类别名称。")
        except Exception as e:
            print(f"❌ 错误: 无法加载类别文件 '{class_names_path}': {e}")
        
        print("------------------------")

    def _preprocess(self, cv2_image):
        """
        将从OpenCV截取的方块图像预处理成模型需要的格式
        """
        # 1. BGR to PIL.Image
        img = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

        # 2. Pad to square and resize
        w, h = img.size
        if w > h: padding = (0, (w - h) // 2)
        else: padding = ((h - w) // 2, 0)
        
        # 使用PIL进行填充和缩放
        from PIL import ImageOps
        img = ImageOps.expand(img, padding)
        img = img.resize((self.target_size, self.target_size))

        # 3. Grayscale
        img = img.convert('L')

        # 4. ToTensor (转换为Numpy数组，并调整维度和范围)
        img_np = np.array(img, dtype=np.float32)
        img_np = img_np / 255.0  # 归一化到 [0, 1]
        
        # 添加批次和通道维度: (H, W) -> (1, 1, H, W)
        input_tensor = np.expand_dims(np.expand_dims(img_np, axis=0), axis=0)
        
        return input_tensor

    def predict(self, tile_image):
        """
        对单个字母方块（BGR格式）进行预测
        """
        if not self.session or not self.class_names:
            return None, 0.0

        # 1. 预处理图像
        input_tensor = self._preprocess(tile_image)

        # 2. 执行推理
        # session.run()返回的是一个列表，因为模型可能有多个输出
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # 3. 后处理结果
        # 获取第一个输出（我们的模型只有一个输出）
        logits = outputs[0][0] # 形状是 (num_classes,)
        
        # 应用Softmax将logits转换为概率
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # 找到置信度最高的类别
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        
        predicted_letter = self.class_names[predicted_idx]
        
        return predicted_letter, confidence * 100
  
# ==============================================================================
#  组件二： 图像分析 (分割、颜色识别)
# ==============================================================================

def find_grid_dynamically(image, debug_mode=False):
    """
    动态地在图像中查找游戏网格
    """
    if image is None: return None, 0

    vis_image = image.copy() if debug_mode else None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 阈值化

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if debug_mode:
        cv2.imwrite("debug_1_threshold.png", thresh)
        print("✅ 已将[步骤1-阈值化结果]保存到 'debug_1_threshold.png'")

    # 2. 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug_mode:
        # 在可视化图上绘制所有找到的初始轮廓（蓝色）
        cv2.drawContours(vis_image, contours, -1, (255, 0, 0), 1) # 蓝色细线
        print(f"  [调试] 步骤2: 找到 {len(contours)} 个初始轮廓。")

    # 3. 筛选方块
    detected_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 使用轮廓面积而非边界矩形面积，更精确
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h)

        if 800 < area < 80000 and 0.75 < aspect_ratio < 1.25:
            detected_boxes.append((x, y, w, h))
            if debug_mode:
                # 将通过筛选的轮廓画成绿色粗框
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        elif debug_mode:
            # 打印被拒绝的轮廓信息，帮助调试
            if area > 100: # 只打印有意义的噪点
                print(f"  [调试] 拒绝轮廓: Area={int(area)}, AspectRatio={aspect_ratio:.2f}")

    if not detected_boxes:
        print("❌ 动态检测：[步骤3] 筛选后未剩下任何方块轮廓。请检查'debug_1_threshold.png'和'debug_2_contours.png'。")
        if debug_mode: cv2.imwrite("debug_2_contours.png", vis_image)
        return None, 0
    
    print(f"  [调试] 步骤3: {len(detected_boxes)} 个轮廓通过筛选。")

    # 4. 组织行列
    detected_boxes.sort(key=lambda b: b[1])
    rows = []
    if detected_boxes:
        current_row = [detected_boxes[0]]; avg_tile_height = detected_boxes[0][3]
        for box in detected_boxes[1:]:
            if box[1] > current_row[0][1] + avg_tile_height * 0.7:
                current_row.sort(key=lambda b: b[0]); rows.append(current_row)
                current_row = [box]
            else: current_row.append(box)
        current_row.sort(key=lambda b: b[0]); rows.append(current_row)

    if not rows:
        print("❌ 动态检测：[步骤4] 无法将方块组织成行。")
        if debug_mode: cv2.imwrite("debug_2_contours.png", vis_image)
        return None, 0
        
    try:
        num_cols = max(len(r) for r in rows)
    except ValueError:
        num_cols = 0 

    final_rows = [row for row in rows if len(row) == num_cols]
    if not final_rows:
        print(f"❌ 动态检测：[步骤4] 所有行都因长度不等于最长行（{num_cols}）而被过滤。")
        if debug_mode: cv2.imwrite("debug_2_contours.png", vis_image)
        return None, 0

    print(f"✅ 动态检测：发现 {len(final_rows)} 行 x {num_cols} 列的网格。")

    if debug_mode:
        for r_idx, row in enumerate(final_rows):
            for c_idx, (x, y, w, h) in enumerate(row):
                 cv2.putText(vis_image, f"R{r_idx+1}C{c_idx+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        debug_path = "debug_2_grid_detection.png"
        cv2.imwrite(debug_path, vis_image)
        print(f"✅ 已将[最终网格检测结果]保存到 '{debug_path}'")

    return final_rows, num_cols


# 定义游戏中各种颜色的 BGR 值
COLOR_GREEN = np.array([100, 170, 106])
COLOR_YELLOW = np.array([88, 180, 201])
COLOR_GRAY = np.array([126, 124, 120])


def analyze_game_board(image_path, predictor, debug_mode=False):
    """
    完整分析游戏截图，返回游戏状态。此版本使用动态网格检测。
    """
    def get_color_details(block):
        # 增加判断，如果方块太小，则取中心区域
        h, w, _ = block.shape
        patch_size = min(10, h//2, w//2)
        start = 5 if patch_size >= 5 else 0
        corner_patch = block[start:start+patch_size, start:start+patch_size]
        if corner_patch.size == 0: return "gray", np.array([0,0,0]), {} # 避免空块错误
        
        avg_color = np.mean(corner_patch, axis=(0, 1))
        distances = {
            "green": np.linalg.norm(avg_color - COLOR_GREEN),
            "yellow": np.linalg.norm(avg_color - COLOR_YELLOW),
            "gray": np.linalg.norm(avg_color - COLOR_GRAY)
        }
        return min(distances, key=distances.get), avg_color, distances

    image = cv2.imread(image_path)
    if image is None: print(f"❌ 错误: 无法加载图片 '{image_path}'"); return None, 0

    detected_rows, detected_word_length = find_grid_dynamically(image, debug_mode=debug_mode)
    
    if not detected_rows:
        print("❌ 错误: analyze_game_board 中断，因为未能自动检测到游戏网格。"); return None, 0

    game_state = []
    print("\n--- 棋盘分析中 ---")

    for r_idx, row in enumerate(detected_rows):
        row_letters = []
        if len(row) != detected_word_length or detected_word_length == 0: continue
            
        for c_idx, (x, y, w, h) in enumerate(row):
            block_original = image[y:y+h, x:x+w]

            if np.mean(block_original) < 30 or np.mean(block_original) > 240: continue

            color, _, _ = get_color_details(block_original)
            
            lower_white = np.array([200, 200, 200]); upper_white = np.array([255, 255, 255])
            mask = cv2.inRange(block_original, lower_white, upper_white)
            binarized_block = np.zeros_like(block_original); binarized_block[mask != 0] = [255, 255, 255]
            letter, conf = predictor.predict(binarized_block)
            
            if letter:
                game_state.append({'letter': letter, 'color': color, 'position': c_idx})
                color_map = {"green": "Gr", "yellow": "Yl", "gray": "Gy"}
                row_letters.append(f" {letter}({color_map.get(color, '?')}) ")
        
        if row_letters: print(f"第 {r_idx+1} 行 (长度 {len(row_letters)}): {''.join(row_letters)}")
            
    if not game_state: print("未在图片中检测到任何已猜测的单词。")
    print("--- 分析完成 ---\n")
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


def main_solver_logic(image_path): 
    """
    主逻辑函数，它执行所有计算并返回结构化数据。
    """
    # 1. 初始化预测器
    predictor = ONNXPredictor()
    if not predictor.session:
        print("❌ 模型加载失败，程序退出。")
        return "CNN模型加载失败", []

    # 2. 分析图像获取游戏状态
    game_state, word_length = analyze_game_board(image_path, predictor, debug_mode=False)
    if not game_state:
        return "💡 未能自动检测到棋盘信息", []

    # 3. 根据检测到的长度选择并加载词库
    wordlist_path = os.path.join('wordlists', f"wordlist_{word_length}.txt")
    try:
        with open(wordlist_path, 'r') as f:
            all_words = [line.strip().upper() for line in f if len(line.strip()) == word_length]
    except FileNotFoundError:
        print(f"❌ 错误: 未找到长度为 {word_length} 的词库 '{wordlist_path}'。")
        return f"缺少词库: wordlist_{word_length}.txt", []
    
    if not all_words:
        print(f"❌ 错误: 在词库中没有找到任何长度为 {word_length} 的单词。")
        return f"词库中缺少长度为 {word_length} 的单词", []
        
    print(f"✅ 已检测到单词长度为 {word_length}，并加载 {len(all_words)} 个对应单词。")

    # 4. 求解
    possible_words = filter_word_list(all_words, game_state)
    
    # 5. 获取建议
    suggestion, possibilities = suggest_best_word(possible_words, all_words)

    return suggestion, possibilities