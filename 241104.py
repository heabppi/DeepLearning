import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import numpy as np

# 1) 총 페이지 개수
response = requests.get('https://www.yes24.com/Product/Search?domain=ALL&query=%25EA%25B0%2595%25ED%2599%2594%25ED%2595%2599%25EC%258A%25B5')

# HTML 받아오기
HTML = response.text
soup = BeautifulSoup(HTML, 'html.parser')

# 마지막 페이지 개수
last_page = int(soup.select_one('a.bgYUI.end').get('title'))

# 빈 데이터프레임 만들기
df = pd.DataFrame(columns=['제목','작가','출판사','가격'])

# 12페이지 도서명 + 저자+ 출판사 + 가격
for page in range(1, last_page+1):
    response = requests.get(f'https://www.yes24.com/Product/Search?domain=ALL&query=%25EA%25B0%2595%25ED%2599%2594%25ED%2595%2599%25EC%258A%25B5&page={page}')

    HTML = response.text
    soup = BeautifulSoup(HTML, 'html.parser')

    # 'li' 태그의 목록을 모두 선택
    book_items = soup.select('#yesSchList > li')

    book_items = len(book_items)
    print(f"{page} page-number : {book_items} books")
    
    for book in range(1, book_items+1):
        book_table_ele = soup.select(f'#yesSchList > li:nth-child({book}) > div > div.item_info > div.info_row.info_name > a.gd_name')
        author_table_ele = soup.select(f'#yesSchList > li:nth-child({book}) > div > div.item_info > div.info_row.info_pubGrp > span.authPub.info_auth > a')
        publisher_table_ele = soup.select(f'#yesSchList > li:nth-child({book}) > div > div.item_info > div.info_row.info_pubGrp > span.authPub.info_pub > a')
        cost_ele = soup.select(f'#yesSchList > li:nth-child({book}) > div > div.item_info > div.info_row.info_price > strong > em')

        book = book_table_ele[0].text.strip() if book_table_ele else np.nan
        author = author_table_ele[0].text.strip() if author_table_ele else np.nan
        publisher = publisher_table_ele[0].text.strip() if publisher_table_ele else np.nan
        cost = cost_ele[0].text.strip() if cost_ele else np.nan

        df.loc[len(df)] = [book, author, publisher, cost]

df