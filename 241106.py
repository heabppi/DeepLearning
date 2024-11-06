# OPEN API 이용하기


# *********유튜브 영상 봇 만들기************

## 하나씩 코드
from youtube_transcript_api import YouTubeTranscriptApi
video_url = 'https://www.youtube.com/watch?v=CHGMgH2dHSg&list=RDNSCHGMgH2dHSg&start_radio=1'

# get video id
def get_video_id(video_url):
    video_id = video_url.split('?v=')[-1][:11]
    return video_id
video_id = get_video_id(video_url)


# transcript language
transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

for transcript in transcript_list:
    print(transcript.language)
    print(transcript.language_code)

# 자막 가져오기
transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])

# 텍스트 형식으로 자막가져오기
from youtube_transcript_api.formatters import TextFormatter
text_fomatter = TextFormatter()
text_transcript = text_fomatter.format_transcript(transcript)
text_transcript

# 줄바꿈 제거
trascript_info = text_transcript.replace('\n', ' ')
trascript_info

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 확인
my_variable = os.getenv('OPENAI_API_KEY')  # .env 파일에 MY_VARIABLE=some_value와 같은 형식으로 정의된 값을 가져옴
print("OPENAI_API_KEY:", my_variable)

# Open API 를 사용
from openai import OpenAI
client = OpenAI()

messages = [
    {'role' : 'system', 'content' : '당신은 유튜브 자막을 요약하는 봇입니다.'},
    {'role' : 'user', 'content' : f"""
            아래 [자막]을 주요 내용을 위주로 요약해줘
     
            [양식]
            주요 내용 1:
            주요 내용 2:
            ...
     
            [자막]
            {trascript_info}
            """}
]
response = client.chat.completions.create(
    model = 'gpt-4o-mini',
    messages= messages
)

response.choices[0].message.content

# print 하기
print(response.choices[0].message.content)

# 답변을 해주는 함수 만들기(내가 해본 것)
from openai import OpenAI
client = OpenAI()

def chat_response():

    messages = [
        {'role' : 'system', 'content' : '당신은 유튜브 내용에 대한 조언이나, 답변을 하는 봇입니다.'},
        {'role' : 'user', 'content' : f"""
                아래 [자막]을 주요 내용에 대한 조언이나, 답변을 해줘
        
                [양식]
                주요 내용 1에 대한 답변:
                주요 내용 2에 대한 답변:
                ...
        
                [자막]
                {trascript_info}
                """}
    ]
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages= messages
    )
    return response.choices[0].message.content

# print 하기
print(response.choices[0].message.content)

# 강사님 ver(함수)
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from openai import OpenAI

def answer_bot(question, video_url):
    client = OpenAI()
    # get video id
    video_id = video_url.split('?v=')[-1][:11]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # 텍스트 형식으로 자막가져오기
    for transcript in transcript_list:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        text_fomatter = TextFormatter()
        text_transcript = text_fomatter.format_transcript(transcript)
        transcript_info = text_transcript.replace('\n', ' ')


    messages = [
        {'role' : 'system', 'content' : '당신은 유튜브 자막을 바탕으로 답변하는 봇입니다.'},
        {'role' : 'user', 'content' : f"""
            '{question}'라는 사용자 질문에 
            아래 [자막]을 참고해서 답변해주세요.
        
            [자막]
            {transcript_info}
            """}
    ]
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages= messages
    )

    return response.choices[0].message.content


# 강사님 함수 출력
video_url = 'https://www.youtube.com/watch?v=7CRPB1vuVAc'
answer_bot('이런 비슷한 음악을 하는 래퍼들을 알려줘', video_url)


#******************텍스트 생성****************


# 응답확인
from openai import OpenAI
client = OpenAI()
# system : 시스템의 역할
# user : 유저의 채팅
# assistant : GPT의 답변
messages = [
    {'role' : 'system', 'content' : 'You are a helpful assistant.'},
    {'role' : 'user', 'content' : '오후 5시에 듣기 좋은 노래 가사를 만들어줘'}
]
response = client.chat.completions.create(
    model = 'gpt-4o-mini',
    messages= messages,
    max_tokens= 1000, 
    temperature= 0.8
)

response.choices[0].message.content

# 이전답변 전달
from openai import OpenAI
client = OpenAI()
# system : 시스템의 역할
# user : 유저의 채팅
# assistant : GPT의 답변
messages = [
    {'role' : 'system', 'content' : '제품 가격을 알려주는 도우미'},
    {'role' : 'user', 'content' : '컴퓨터의 가격은?'},
    {'role' : 'assistant', 'content' : '컴퓨터은 1000000원입니다.'},
    {'role' : 'user', 'content' : '모니터의 가격은?'},
    {'role' : 'assistant', 'content' : '컴퓨터은 200000원입니다.'},
    {'role' : 'user', 'content' : '컴퓨터의 가격은 얼마야?'}
]
response = client.chat.completions.create(
    model = 'gpt-4o-mini',
    messages= messages,
    max_tokens= 1000
)

response.choices[0].message.content

# 질문에 답변하는 함수
from openai import OpenAI

def chat_gpt(prompt):
    client = OpenAI()
    messages = [
        {'role' : 'system', 'content' : 'You are a helpful assistant.'},
        {'role' : 'user', 'content' : prompt},
    ]
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages= messages,
        max_tokens= 1000
    )
    return response.choices[0].message.content

# 답변
chat_gpt('오늘 기분은 어때?')

# 실습(챗봇 기능 구현 : 대화 구현)
from openai import OpenAI

def chat_bot(prompt, messages):
    client = OpenAI()
    
    messages.append({'role' : 'user', 'content' : prompt})
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages= messages
    )
    
    gpt_response = response.choices[0].message.content
    messages.append({'role' : 'assistant', 'content' : gpt_response})
    return gpt_response, messages

messages = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]

while True:
    prompt = input('질문을 입력하세요.(종료 하려면 q 입력)')
    print(f'User : {prompt}')
    if prompt == 'q':
        break
    gpt_response, messages = chat_bot(prompt, messages)
    print(f'GPT : {gpt_response}')


# ************음성, 텍스트, 이미지, 이미지 캡셔닝 생성**********


# 이미지 생성
from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="imabe.jpg",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url


# Text to Speech
# voice = (alloy, echo, fable, onyx, nova, and shimmer)
from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="nova",
    input="이것은 음성 녹음 테스트를 위한 문장입니다. 잘 들리나요?",
)

response.stream_to_file("output.mp3")

# Speech to Text
from openai import OpenAI
client = OpenAI()

audio_file = open("audio.m4a", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)
print(transcript .text)

# GPT vision
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])

# 이미지 파일
import base64
from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "cat.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image aswer?",
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/jpeg;base64,{base64_image}"
          },
        },
      ],
    }
  ],
)

print(response.choices[0].message.content)

#*********영상 나래이션******************

# 1) 필요한 라이브버리 가져오기
from IPython.display import display, Image, Audio

import cv2 
import base64
import time
from openai import OpenAI
import os
import requests

client = OpenAI()

# 2) 영상에서 프레임을 활용하여 이미지를 가져옵니다.
video = cv2.VideoCapture("귀여운 강아지들.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")


# 3) 이미지로부터 텍스트를 가져오기(이미지 -> 텍스트)
PROMPT_MESSAGES = [
    {
        "role": "user",
        ### 아래 content안의 내용을 수정해보세요
        "content": [
            """
            업로드하려는 동영상의 프레임입니다. 동영상과 함께 업로드할 수 있는 설득력 있는 설명을 생성합니다. 
            영상하이라이트는 생략하고 설명만 생성해주세요.
            업로드하는 영상은 귀여운 동물영상입니다. 귀여워 하는 추임새도 포함해주세요.
            """,
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
        ],
    },
]
params = {
    "model": "gpt-4o-mini",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 100,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)

# 4) 생성된 텍스트로부터 오디오 가져오기(텍스트 -> 오디오)
response = requests.post(
    "https://api.openai.com/v1/audio/speech",
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    },
    json={
        "model": "tts-1-1106",
        "input": result.choices[0].message.content,
        "voice": "onyx",
    },
)

audio = b""
for chunk in response.iter_content(chunk_size=1024 * 1024):
    audio += chunk
Audio(audio)

with open("output.mp3", "wb") as file:
    file.write(audio)


#**********임베딩 모델 이해하기*******************
from openai import OpenAI
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

cat = get_embedding('고양이')
dog = get_embedding('강아지')
din = get_embedding('공룡')
bui = get_embedding('건물')
car = get_embedding('자동차')
coke = get_embedding('콜라')

import pandas as pd
df = pd.DataFrame({"cat" : cat, "dog" : dog, "din" :din})
df.T


from sklearn.decomposition import PCA

# PCA 적용 (주성분 수 지정, 예: 2)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df.T)

# 결과를 데이터프레임으로 변환
principal_df = pd.DataFrame(data=principal_components, columns=['X', 'Y'], index = df.columns)
principal_df.head()

import plotly.express as px
fig = px.scatter(principal_df, x="X", y="Y", hover_name=principal_df.index)
fig.show()


# PDF 요약하기(실습)
from PyPDF2 import PdfReader

pdf_file = '20241105_company_62797000.pdf' # PDF 파일 경로
reader = PdfReader(pdf_file) # PDF 문서 읽기
meta = reader.metadata   # PDF 문서의 메타데이터 가져옴(문서 제목, 저자 등)
page = reader.pages[0]   # 첫 페이지 내용 가져옴 
page_text = page.extract_text() # 페이지의 텍스트 추출 

print("- 문서 제목:", meta.title)
print("- 문서 저자:", meta.author)
print("- 첫 페이지 내용 일부:\n",  page_text[:300])

import textwrap
shorten_summary = textwrap.shorten(page_text, 250, placeholder=' [..이하 생략..]')