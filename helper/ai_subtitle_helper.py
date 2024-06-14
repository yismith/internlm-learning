import json
import os

if __name__ == '__main__':

    # 课程名称
    # course_name = '01-introduction'
    # course_name = '02-helloworld'
    # course_name = '03-huixiangdou'
    # course_name = '04-xtuner'
    course_name = '05-lmdeploy'

    # 定义文件路径
    json_file_path = os.path.join(os.path.dirname(__file__), '..', 'camp2', 'assets', 'video', 'aisubtitle',
                                  '%s.json' % course_name)
    txt_file_path = os.path.join(os.path.dirname(__file__), '..', 'camp2', 'assets', 'video', 'aisubtitle',
                                 '%s.txt' % course_name)

    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取并拼接 content 字段
    content_list = [entry['content'] for entry in data['body']]
    concatenated_content = ''.join(content_list)

    # 将拼接好的内容写入 txt 文件
    with open(txt_file_path, 'w', encoding='utf-8') as file:
        file.write(concatenated_content)

    print(f"拼接后的内容已写入到 {txt_file_path}")

