# text
# streamlit_text.py
import streamlit as st

# 이모지: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title('타이틀 with 이모지 :bar_chart:')

st.header('이 글자는 헤더입니다.')
st.subheader('이 글자는 서브헤더입니다.')
st.caption('이 글자는 캡션입니다.')
st.text('일반 텍스트 입니다.')
code = '''
def sample_function():
    print('이것은 샘플 함수입니다.')
'''
st.code(code, language = 'python')
st.title('')
st.title('마크 다운 지원')
st.markdown('# 타이틀')
st.markdown('## 헤더')
st.markdown('### 서브헤더')
st.markdown('*이탤릭체*')
st.markdown('_이탤릭체_')
st.markdown('**볼드체**')
st.markdown('* 목록1')
st.markdown('+ 목록2')
st.markdown('- 목록3')
st.markdown('1. 번호 목록1')
st.markdown('2. 번호 목록2')
st.markdown('> 인용문 작성')
st.markdown('[Google](https://www.google.com)')
st.markdown('---')
st.markdown('***')
st.markdown(':green[녹색], :red[빨강]')

st.latex(r'\sqrt{x^2+y^2}=1')

## 데이터프레임 다루기
import streamlit as st
import pandas as pd


st.title('데이터프레임 튜토리얼')

# 예시 데이터 생성
data = {
    "회사명": ["삼성전자", "현대자동차", "SK하이닉스", "NAVER", "LG화학"],
    "주식 코드": ["005930", "005380", "000660", "035420", "051910"],
    "현재 가격": [56000, 180000, 85000, 350000, 680000],
    "전일 대비": [-500, 2000, -1500, 5000, 8000],
    "시가": [56500, 178000, 86500, 345000, 672000],
    "고가": [57000, 182000, 87000, 352000, 690000],
    "저가": [55000, 176000, 84000, 340000, 670000],
    "거래량": [10500000, 1400000, 750000, 500000, 800000]
}

# 데이터프레임 생성
df = pd.DataFrame(data)
st.dataframe(df)
st.metric(label = '삼성전자', value = 56000, delta = -500)
st.metric(label = '현대자동차', value = 180000, delta = 2000)

col1, col2, col3 = st.columns(3)
col1.metric(label = df.loc[0, '회사명'],
             value = str(df.loc[0, '현재 가격']),
               delta = str(df.loc[0, '전일 대비']))
col2.metric(label = df.loc[1, '회사명'],
             value = str(df.loc[1, '현재 가격']),
               delta = str(df.loc[1, '전일 대비']))
col3.metric(label = df.loc[2, '회사명'],
             value = str(df.loc[2, '현재 가격']),
               delta = str(df.loc[2, '전일 대비']))
col2.metric(label = df.loc[3, '회사명'],
             value = str(df.loc[3, '현재 가격']),
               delta = str(df.loc[3, '전일 대비']))

## 차트 다루기
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title('차트 튜토리얼')

# 예시 데이터 생성
students = ['Student A', 'Student B', 'Student C', 'Student D', 'Student E', 
            'Student F', 'Student G', 'Student H', 'Student I', 'Student J']
math_scores = [85, 78, 92, 76, 89, 65, 95, 88, 70, 82]
science_scores = [90, 80, 88, 70, 92, 60, 96, 85, 72, 78]

fig, ax = plt.subplots()
plt.scatter(math_scores, science_scores, alpha= 0.6)
st.pyplot(fig)

fig2 = go.Figure(data=go.Scatter(x=math_scores, y=science_scores,mode='markers', name='markers'))
st.plotly_chart(fig2)


## UI
import streamlit as st
import pandas as pd
from datetime import datetime as dt
import datetime

st.subheader('버튼')
# 버튼 클릭
button = st.button('버튼입니다~')

if button:
    st.write('버튼이 눌러졌습니다.')

st.subheader('체크박스')
check_box = st.checkbox("체크 하실래요?")

if check_box:
    st.write('체크 하셨습니다.')

st.subheader('')
st.subheader('radio')
radio = st.radio(
    '당신의 성별은 무엇입니까?',
    ('남자', '여자')
)

if radio == '남자':
    st.write('남자를 선택하셨습니다.')
elif radio == '여자':
    st.write('여자를 선택하셨습니다.')
else:
    st.write('')


st.subheader('')
st.subheader('multiselect')
multiselect = st.multiselect(
    '몇 번을 선택하시겠습니까?',
    [1, 2, 3, 4, 5]
)

st.write(f'{multiselect}를 선택하셨습니다.')

st.subheader('')
st.subheader('slider')
slider = st.slider(
    '범위를 지정할 수 있습니다.',
    0, 100, (33, 66)
)
st.write(f'선택범위 :{slider}')

st.subheader('')
st.subheader('slider(date)')
slider_date = st.slider(
    label = '날짜를 선택해주세요.',
    min_value = dt(2023, 1, 1,0,0),
    max_value = dt(2023, 12, 31,23,59),
    value = dt(2023, 12, 21, 16, 37),
    step = datetime.timedelta(hours=1),
    format = 'yyyy/DD/MM HH:mm')
st.write(slider_date)

st.subheader('')
st.subheader('text input')
text_input = st.text_input(
    label = '이름이 무엇인가요?',
    placeholder = '이름을 입력해주세요.'
)
if text_input:
    st.write(f'당신의 이름은 {text_input}입니다.')

st.subheader('')
st.subheader('number input')
num = st.number_input(
    label = '나이를 입력해주세요.',
    step = 1, # step
    min_value = 20,  # 최소 허용 값
    max_value = 100, # 최대 허용 값
    value = 40 # 시작 위치
)

st.write(f'당신의 나이는 {num}살 입니다.')

st.subheader('')
st.subheader('expander(더보기)')
with st.expander("더 보기"):
    st.write('섹션1')
    st.write('섹션2')

# 탭 활용
import streamlit as st
import pandas as pd
from datetime import datetime as dt
import datetime

st.subheader('버튼')
# 버튼 클릭
button = st.button('버튼입니다~')

if button:
    st.write('버튼이 눌러졌습니다.')

st.subheader('체크박스')
check_box = st.checkbox("체크 하실래요?")

if check_box:
    st.write('체크 하셨습니다.')

st.subheader('')
st.subheader('radio')
radio = st.radio(
    '당신의 성별은 무엇입니까?',
    ('남자', '여자')
)

if radio == '남자':
    st.write('남자를 선택하셨습니다.')
elif radio == '여자':
    st.write('여자를 선택하셨습니다.')
else:
    st.write('')


st.subheader('')
st.subheader('multiselect')
multiselect = st.multiselect(
    '몇 번을 선택하시겠습니까?',
    [1, 2, 3, 4, 5]
)

st.write(f'{multiselect}를 선택하셨습니다.')

st.subheader('')
st.subheader('slider')
slider = st.slider(
    '범위를 지정할 수 있습니다.',
    0, 100, (33, 66)
)
st.write(f'선택범위 :{slider}')

st.subheader('')
st.subheader('slider(date)')
slider_date = st.slider(
    label = '날짜를 선택해주세요.',
    min_value = dt(2023, 1, 1,0,0),
    max_value = dt(2023, 12, 31,23,59),
    value = dt(2023, 12, 21, 16, 37),
    step = datetime.timedelta(hours=1),
    format = 'yyyy/DD/MM HH:mm')
st.write(slider_date)

st.subheader('')
st.subheader('text input')
text_input = st.text_input(
    label = '이름이 무엇인가요?',
    placeholder = '이름을 입력해주세요.'
)
if text_input:
    st.write(f'당신의 이름은 {text_input}입니다.')

st.subheader('')
st.subheader('number input')
num = st.number_input(
    label = '나이를 입력해주세요.',
    step = 1, # step
    min_value = 20,  # 최소 허용 값
    max_value = 100, # 최대 허용 값
    value = 40 # 시작 위치
)

st.write(f'당신의 나이는 {num}살 입니다.')

st.subheader('')
st.subheader('expander(더보기)')
with st.expander("더 보기"):
    st.write('섹션1')
    st.write('섹션2')

## 사이드 바 활용
import streamlit as st

st.title('사이드바 활용하기')

page = st.sidebar.selectbox('페이지 선택',
                              ['Home', '데이터분석','시각화'])

if page == 'Home':
    st.title('여기는 Home 입니다.')
elif page == '데이터분석':
    st.title('여기는 데이터분석 페이지 입니다.')
elif page == '시각화':
    st.title('여기는 시각화 페이지 입니다.')

## 이미지, 오디오, 비디오 다루기
import streamlit as st
from PIL import Image

st.title('이미지 다루기')
st.subheader('이미지 다루기')

# 로컬 이미지 활용하기
image_local = Image.open('file/img.webp')
st.image(image_local, width = 400, caption = '사람들')
image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg'
st.image(image_url, width = 400, caption = '풍경')


## 실습( 이미지 생성 웹앱 만들기)
import streamlit as st
from openai import OpenAI

client = OpenAI()
st.title('이미지 생성기')

prompt = st.text_input(
    label = '무슨 이미지를 생성하시겠어요?',
    placeholder = '생성 할 이미지의 프롬프르를 입력해주세요.'
)
if prompt:
    response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
    )

    image_url = response.data[0].url
    st.image(image_url, width = 400)

## 오디오, 비디오 다루기
import streamlit as st

st.title('오디오, 비디오 다루기')

st.subheader('오디오 다루기')
audio_file = open('file/output.mp3', 'rb')
st.audio(audio_file, format = 'audio/mp3')

st.subheader('비디오 다루기')
video_file = 'file/bison.mp4'
st.video(video_file)

## PDF 요약하기
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
client = OpenAI()

# 요약 기능
def summarize_text(text):
    messages = [
        {'role' : 'system', 'content' : '당신은 기업리포트를 요약하는 전문가 입니다.'},
        {'role' : 'user', 'content' : f'''
         아래 [기업 리포트]를 한국어로 요약해주세요.

         [기업 리포트]
         {text}
         '''}
    ]
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages= messages,
        max_tokens= 2000, 
        temperature= 0.5
    )
    return response.choices[0].message.content


# Streamlit 페이지 구성
pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

button = st.button('PDF 요약하기')
if button:
    reader = PdfReader(pdf_file) # PDF 문서 읽기

    # 각 페이지를 요약
    summaraize_result_list = []
    for page in reader.pages: 
        page_text = page.extract_text() # 페이지의 텍스트 추출 
        summaraize_result = summarize_text(page_text)
        summaraize_result_list.append(summaraize_result)

    # 각 요약된 결과를 연결 후 요약 진행
    result = " ".join(summaraize_result_list)
    summaraize_all_pages = summarize_text(result)
    st.write(summaraize_all_pages)

# 실습 유튜브 영상 요약하기
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from openai import OpenAI
client = OpenAI()

# 요약 기능
def summarize_text(text):
    messages = [
        {'role' : 'system', 'content' : '당신은 뉴스를 요약하는 전문가 입니다.'},
        {'role' : 'user', 'content' : f'''
         아래 [뉴스]를 한국어로 요약해주세요.

         [뉴스]
         {text}
         '''}
    ]
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages= messages,
        max_tokens= 2000, 
        temperature= 0.5
    )
    return response.choices[0].message.content


# Streamlit 페이지 구성
video_url = st.text_input("유튜브 URL을 입력하세요.")
button = st.button('유튜브 요약하기')
if button:
    def get_video_id(video_url):
        video_id = video_url.split('?v=')[-1][:11]
        return video_id
    video_id = get_video_id(video_url)
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])

    text_formatter = TextFormatter() # Text 형식으로 출력 지정
    text_formatted = text_formatter.format_transcript(transcript)

    st.write(summarize_text(text_formatted))