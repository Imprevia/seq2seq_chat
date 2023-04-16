from collections import Counter

import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def get_words(txt):
    seg_list = []

    words = jieba.cut(txt)

    for work in words:
        seg_list.append(work)

    c = Counter()  # 计数器
    for x in seg_list:
        if len(x) > 1 and x not in ("\r\n"):
            c[x] += 1  # 个数加一
    return c.most_common(305)

def draw_wordcloud(c):
    cloud = WordCloud(
        # 设置字体，不指定就会出现乱码，文件名不支持中文
        font_path=r"simhei.ttf",
        # font_path=path.join(d,'simsun.ttc'),
        # 设置背景色，默认为黑，可根据需要自定义为颜色
        background_color='white',
        # 词云形状，
        # mask=color_mask,
        # 允许最大词汇
        max_words=300,
        # 最大号字体，如果不指定则为图像高度
        max_font_size=80,
        # 画布宽度和高度，如果设置了msak则不会生效
        width=600,
        height=400,
        margin=2,
        # 词语水平摆放的频率，默认为0.9.即竖直摆放的频率为0.1
        prefer_horizontal=0.8
        # relative_scaling = 0.6,
        # min_font_size = 10
    ).generate_from_frequencies(c)
    plt.imshow(cloud)
    plt.axis("off")
    plt.show()

if __name__=="__main__":
    xls = pd.read_excel(r'train.xlsx', header=0)

    list = []

    for i in range(len(xls)):
        list.append(str(xls["input"][i]))
        list.append(str(xls["output"][i]))

    c = get_words("".join(list))
    dict = {}
    for i in c:
        dict[i[0]] = i[1]

    draw_wordcloud(dict)