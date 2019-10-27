import os, glob
import pandas as pd

df = pd.read_csv("hindawi_insight - metadata.csv")


files = glob.glob("Hindawi_final_text/*/*/*_meta.txt")
for fil in files:
    f = open(fil,"r", encoding="utf-8") 
    idd = fil.split('/')[2]
    id = int(idd)
    l = f.readlines()
    url_i = l.index("الرابط:\n")
    title_i = l.index("اسم الكتاب:\n")
    category_i = l.index("التصنيف:\n")
    author_i = l.index("الكاتب:\n")

    url = l[url_i+1]
    url = url.strip()

    title = l[title_i+1]
    title = title.strip()

    category = l[category_i+1]
    category = category.strip()

    author = l[author_i+1]
    author = author.strip()
    df = df.append(pd.Series([idd, title, author, category, url], index=df.columns ), ignore_index=True)


export_csv = df.to_csv ('newfile.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

