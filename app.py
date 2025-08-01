from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import sys
import io
from solver_logic import main_solver_logic

# --- Flask 配置 ---
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 # 限制上传大小为4MB
app.secret_key = 'super-secret-key' # 用于flash消息

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- REPLACE THE `upload_file` FUNCTION in app.py ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 在保存文件前，确保目标文件夹存在
            upload_dir = os.path.dirname(filepath)
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            file.save(filepath)
            
            # 1. 捕获print输出作为日志
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            # 2. 调用主逻辑函数
            suggestion, possibilities = main_solver_logic(image_path=filepath)
            
            # 3. 恢复标准的stdout
            sys.stdout = old_stdout
            analysis_log = captured_output.getvalue()

            # 4. 将所有需要的数据传递给模板
            return render_template('index.html', 
                                   image_url=filepath,
                                   suggestion=suggestion,
                                   possibilities=possibilities,
                                   analysis_log=analysis_log)

    # 如果是GET请求，只显示上传页面
    return render_template('index.html')

if __name__ == '__main__':
    # 确保uploads文件夹存在
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)