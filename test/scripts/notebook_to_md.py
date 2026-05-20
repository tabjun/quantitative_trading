import json
import base64
import os
import sys

def extract_notebook_to_md(ipynb_path, output_md_path):
    """
    주피터 노트북 파일을 읽어 출력 결과(텍스트, 이미지)를 마크다운 파일로 변환합니다.
    """
    if not os.path.exists(ipynb_path):
        print(f"Error: 파일을 찾을 수 없습니다 - {ipynb_path}")
        return

    # 이미지 저장 폴더 결정 (test/images가 존재하면 최우선으로 사용하여 프로젝트 일관성 유지)
    workspace_test_images = r"C:\Users\jun99\OneDrive\바탕 화면\Analysis\toy_agent_project\quantitative_trading\test\images"
    if os.path.exists(workspace_test_images):
        image_dir = workspace_test_images
    else:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(output_md_path)))
        test_images_dir = os.path.join(parent_dir, "images")
        if os.path.exists(test_images_dir):
            image_dir = test_images_dir
        else:
            image_dir = os.path.join(os.path.dirname(output_md_path), "images")
            
    os.makedirs(image_dir, exist_ok=True)
    
    # 노트북 파일명 기반 이미지 접두사
    base_name = os.path.splitext(os.path.basename(ipynb_path))[0]

    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    md_content = []
    md_content.append(f"# 📊 Notebook Analysis Report: {base_name}\n")
    md_content.append(f"*Auto-generated from {os.path.basename(ipynb_path)}*\n\n---\n")

    image_counter = 1

    for i, cell in enumerate(notebook.get('cells', [])):
        if cell['cell_type'] == 'markdown':
            # 마크다운 셀은 그대로 추가
            md_content.append("".join(cell.get('source', [])) + "\n\n")
            
        elif cell['cell_type'] == 'code':
            # 코드 셀의 소스 코드는 옵션 (현재는 출력만 중점적으로 다룸)
            # source_code = "".join(cell.get('source', []))
            # md_content.append(f"```python\n{source_code}\n```\n")
            
            outputs = cell.get('outputs', [])
            if not outputs:
                continue
                
            md_content.append(f"### Cell {i+1} Output\n")
            
            for output in outputs:
                # 1. 일반 텍스트 출력 (print 문 등)
                if output['output_type'] == 'stream':
                    text = "".join(output.get('text', []))
                    md_content.append(f"```text\n{text}\n```\n")
                
                # 2. 실행 결과 또는 디스플레이 데이터 (그래프, 표 등)
                elif output['output_type'] in ['execute_result', 'display_data']:
                    data = output.get('data', {})
                    
                    # 텍스트 결과가 있는 경우
                    if 'text/plain' in data:
                        text = "".join(data['text/plain'])
                        md_content.append(f"```text\n{text}\n```\n")
                        
                    # HTML 결과 (Pandas DataFrame 등)
                    if 'text/html' in data:
                        html = "".join(data['text/html'])
                        # HTML은 마크다운에 그대로 삽입 가능하지만 깔끔함을 위해 블록 처리
                        md_content.append(f"<details><summary>View Table/HTML</summary>\n\n{html}\n\n</details>\n")
                    
                    # 이미지 결과 (Matplotlib 그래프 등)
                    if 'image/png' in data:
                        img_data = data['image/png']
                        img_bytes = base64.b64decode(img_data)
                        
                        img_filename = f"{base_name}_plot_{image_counter}.png"
                        img_filepath = os.path.join(image_dir, img_filename)
                        
                        with open(img_filepath, 'wb') as img_f:
                            img_f.write(img_bytes)
                            
                        # 마크다운에 이미지 링크 추가 (상대 경로)
                        rel_img_path = os.path.relpath(img_filepath, os.path.dirname(output_md_path)).replace("\\", "/")
                        md_content.append(f"![Plot {image_counter}]({rel_img_path})\n\n")
                        image_counter += 1

    # 최종 마크다운 파일 저장
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("".join(md_content))
        
    print(f"✅ 변환 완료! 보고서가 저장되었습니다: {output_md_path}")
    print(f"✅ 추출된 이미지 수: {image_counter - 1}개 (저장 위치: {image_dir})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python notebook_to_md.py <입력_노트북.ipynb> [출력_마크다운.md]")
        sys.exit(1)
        
    ipynb_file = sys.argv[1]
    
    # 출력 파일명이 제공되지 않은 경우, 입력 파일명 기반으로 자동 생성
    if len(sys.argv) >= 3:
        md_file = sys.argv[2]
    else:
        md_file = os.path.splitext(ipynb_file)[0] + "_result.md"
        
    extract_notebook_to_md(ipynb_file, md_file)
