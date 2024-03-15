import requests
import json
import pandas as pd
from datetime import datetime, timedelta

KEY = 'aaaa'
ITEM_N = 10 # 조회일 출력할 영화개수 
STARTDATE = datetime(2017, 1, 1)  # 조회 시작 날짜
ENDDATE = datetime(2023, 10, 20)  # 조회 끝나는 날짜
NATION = "K"  # 한국 영화
MULTI = "N"  # 상업영화 만  

movie_list = []

# 날짜 범위 생성
current_date = STARTDATE
while current_date <= ENDDATE:
    DATE = current_date.strftime('%Y%m%d')
    # print(DATE)
    try:
        url = f'http://kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?key={KEY}&repNationCd={NATION}&multiMovieYn={MULTI}&targetDt={DATE}&itemPerPage={ITEM_N}'
        res = requests.get(url)
        text = res.text

        d = json.loads(text)

        for b in d['boxOfficeResult']['dailyBoxOfficeList']:
            rank = b['rank']
            title = b['movieNm']
            movieid=b['movieCd']
            audience = b['audiCnt']
            total_audience = b['audiAcc']
            sales = b['salesAmt']
            total_sales = b['salesAcc']
            release_date = b['openDt']

            movie_list.append([DATE, rank, title, movieid, audience, total_audience, sales, total_sales, release_date])
    except:
        print(f'error : {DATE}')
    
    # 현재 날짜를 1일 증가시킴
    current_date += timedelta(days=1)

# 데이터 프레임 생성 및 열 이름 지정
data = pd.DataFrame(movie_list, columns=['날짜', '순위', '영화제목', '영화코드','관객수', '누적관객수', '매출액', '누적매출액', '개봉일'])
print(f'최초 df Info: {data.info()}')
# '날짜' 열과 '개봉일' 열을 datetime 형식으로 변환
data['날짜'] = pd.to_datetime(data['날짜'], format='%Y%m%d', errors='coerce') # 날짜가 아닌 값이 있는 행은 NaN
data['개봉일'] = pd.to_datetime(data['개봉일'], errors='coerce') #날짜가 아닌 값이 있는 행은 NaN

# 유효한 날짜가 아닌 행을 제거
data = data.dropna(subset=['날짜', '개봉일'])
print(f'최종 df Info: {data.info()}')

# 개봉후 N일 칼럼 추가 
data['개봉후N일'] = (data['날짜'] - data['개봉일']).dt.days
data[['순위','관객수', '누적관객수', '매출액', '누적매출액', '개봉후N일']]=data[['순위','관객수', '누적관객수', '매출액', '누적매출액', '개봉후N일']].astype(int)

data.to_csv("./201601-202310_movielist.csv", mode='w', encoding='utf-8', index=False)