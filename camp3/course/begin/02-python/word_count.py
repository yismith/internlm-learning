import re


def wordcount(text):
    # 将文本转换为小写，以确保计数时不区分大小写
    text = text.lower()
    # 使用正则表达式替换标点符号为空格
    text = re.sub(r'[^\w\s]', ' ', text)
    # 分割文本为单词列表
    words = text.split()
    # 创建一个空字典来存储单词计数
    word_count = {}
    # 遍历单词列表，统计每个单词出现的次数
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count


def main():
    texts = [
        """Hello world!  
This is an example.  
Word count is fun.  
Is it fun to count words?  
Yes, it is fun!""",
        """
Got this panda plush toy for my daughter's birthday,
who loves it and takes it everywhere. It's soft and
super cute, and its face has a friendly look. It's
a bit small for what I paid though. I think there
might be other options that are bigger for the
same price. It arrived a day earlier than expected,
so I got to play with it myself before I gave it
to her.
    """
    ]

    for text in texts:
        print(text)
        print(wordcount(text))


if __name__ == '__main__':
    main()
