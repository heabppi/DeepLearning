# 실습(직접 한 것)

# 1. 유튜브 인공지능 검색 URL에 접속합니다.
# 2. 상위 5개의 영상에 대해서 영상 댓글 가져오기 (스크롤 5번)
#     1. 광고 영상, 쇼츠는 제외
# 3. 아래 표로 결과 작성


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd

df = pd.DataFrame(columns=['video_title', 'video_comment'])

driver = webdriver.Chrome()
driver.get('https://www.youtube.com/results?search_query=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5')
driver.maximize_window()
time.sleep(2)


video_list = driver.find_elements(By.CSS_SELECTOR, '#dismissible > div.text-wrapper.style-scope.ytd-video-renderer')

for video in video_list[:5]:
    video_title = video.find_element(By.CSS_SELECTOR, '#video-title > yt-formatted-string.style-scope.ytd-video-renderer').text
    video.click()
    time.sleep(1)

    body = driver.find_element(By.CSS_SELECTOR, 'body')
    for i in range(5):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
    
    comments = driver.find_elements(By.CSS_SELECTOR, '#content-text > span.yt-core-attributed-string.yt-core-attributed-string--white-space-pre-wrap')
    for comment in comments:
        video_comment = comment.text
        df.loc[len(df)] = [video_title, video_comment]
        time.sleep(0.3)

    driver.back()
    time.sleep(0.5)

driver.quit()
print(df)