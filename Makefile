# Makefile示例

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

# 清理缓存等临时文件
clean:
	find . -type f -name "*.pyc" -delete
	rm -rf __pycache__
