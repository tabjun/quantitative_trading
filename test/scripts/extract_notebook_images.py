import json
import os
import base64
import sys

def extract_images_from_notebook(notebook_path, output_dir):
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)
        
    image_counter = 1
    extracted_count = 0
    
    # 노트북 파일 이름 확인을 통해 접두사 결정
    base_name = os.path.basename(notebook_path)
    is_multi_ticker = "3_time_series_multi_ticker_test" in base_name
    
    # 노트북의 셀을 순회하며 이미지 출력(image/png)을 추출합니다.
    for cell in nb_data.get('cells', []):
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                data = output.get('data', {})
                if 'image/png' in data:
                    img_data_base64 = data['image/png']
                    # 줄바꿈 제거
                    if isinstance(img_data_base64, list):
                        img_data_base64 = "".join(img_data_base64)
                    else:
                        img_data_base64 = img_data_base64.replace('\n', '')
                        
                    img_bytes = base64.b64decode(img_data_base64)
                    
                    # 파일명 형식 유연화 분기
                    if is_multi_ticker:
                        output_file = os.path.join(output_dir, "3_multi_ticker_best_model_forecast.png")
                    else:
                        output_file = os.path.join(output_dir, f"2_time_series_advance_test_plot_{image_counter}.png")
                        
                    with open(output_file, 'wb') as img_f:
                        img_f.write(img_bytes)
                        
                    print(f"Saved: {output_file}")
                    image_counter += 1
                    extracted_count += 1
                    
    print(f"Extraction completed: {extracted_count} images saved to {output_dir}")

if __name__ == "__main__":
    # 실행 시 인자로 노트북 경로를 받거나 기본 경로 사용
    notebook_path = "test/models/2_time_series_advance_test.ipynb"
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
        
    output_dir = "test/images"
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        
    extract_images_from_notebook(notebook_path, output_dir)

