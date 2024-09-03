[<img src="https://img.shields.io/badge/notion-00AA55?style=for-the-badge&logo=notion&logoColor=white"/>](https://www.notion.so/QnA-2f31917289ca4f5fa4436dedbecfa00e?pvs=4)

## 주제
재경 업무 지원 QnA 서비스

## 목표
재경 직무 질의응답 Dataset 을 LLM을 활용하여 내외부 문서 기반 질의 응답 시스템을 만들어 현직 재경 업무 담당자의 올바른 의사 결정을 도와주는 웹 서비스를 구현한다.

<br/>

## 개요
1. 재경 직무 관련 기출문제를 pdf 형식으로 크롤링하고, pdf 파일을 json 파일로 변환한다. 추가 데이터셋은 2020-2022년 기출문제를 바탕으로 재경 직무 현직자들이 실제 질문할만한 질의응답 내용으로 자체적으로 변형한 후 json 파일로 작성한다.
2. 준비된 데이터셋을 토크나이징 등 데이터 전처리 과정을 통하여 가공한다.
3. 전처리가 완료된 데이터셋을 준비된 모델에 학습시킨다.
4. 학습이 완료된 모델의 학습 결과를 테스트한다.
5. 결과물을 스트림릿으로 표현한다.
   
<br/>

## 데이터셋 선정 배경
현직 회계사들이 기업의 회계 감사를 할 때 내부 재경 직무 담당자들의 빈번하고 사소한 실수를 발견하여 감사를 진행하는데 차질이 생겨 불편하다는 불만사항이 있었다. 재경 직무 수행 시 발생할 수 있는 실수를 최소화할 수 있는 웹 서비스의 개발로 올바른 의사 결정을 하는데 도움이 되고자 하여 선정하게 되었다. 

<br/>

## Team 2
이재원(PM) 정영재 전상욱 정우영

<br/>

## Stack
##### 개발환경
<img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"/><img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">  

<br/>

##### 데이터 수집 및 처리
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"/><img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"><img src="https://img.shields.io/badge/pytorch-FFFB54?style=for-the-badge&logo=pytorch&logoColor=white"><img src="https://img.shields.io/badge/openai-506065?style=for-the-badge&logo=openai&logoColor=white"><img src="https://img.shields.io/badge/llama3-30C02B?style=for-the-badge&logo=llama3&logoColor=white">

<br/>

##### 웹페이지로 시각화
<img src="https://img.shields.io/badge/Streamlit-43B02A?style=for-the-badge&logo=Streamlit&logoColor=white">  

<br/>

## 모델 학습 결과
- 첫 번째 시도(Llama3)
  ![image (2)](https://github.com/user-attachments/assets/a4e24b2d-528f-4eb3-85fd-9ef6df492bed)

- 두 번째 시도(Llama3)
  ![image (1)](https://github.com/user-attachments/assets/d700bc84-5afb-407a-926a-be65ae6c5adf)

- 세 번째 시도(OpenAI)
  ![image (3)](https://github.com/user-attachments/assets/d5263ee3-75b6-4532-a71d-21546244ab68)

<br/>

## Streamlit 구현



<br/>

## 한계 및 개선점
- 학습은 성공적이었지만 시간부족으로 인해 스트림릿으로 구현까지 수행하지는 못함.<br/>
- 국제회계기준(IFRS)이라는 대원칙이 있지만, 실제 기업별 세부 회계 정책이 각각 약간씩 상이하므로 각 기업에 맞추어 파인튜닝 필요.

<br/>
