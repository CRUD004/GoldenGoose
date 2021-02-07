from datetime import time, datetime
from enum import Enum

import sys
import os
import time
import win32com.client

import pandas as pd


class reqType(Enum):
    ORDER = 0
    SISE = 1


def waitReqLimit(reqType):
    remainCount = obj_CpStatus.GetLimitRemainCount(reqType.value)

    if remainCount > 0:
        # print('남은 횟수: ', remainCount)
        return True
    
    remainTime = obj_CpStatus.LimitRequestRemainTime
    print('조회 제한 회피 time wait %.2f초' % ((remainTime/1000.0)+1.0))
    time.sleep((remainTime/1000.0)+1.0)

    return True



def call_CpSvr7254(code, data_count, start_date):

    obj_CpSvr7254.SetInputValue(0, code) #string 종목코드
    obj_CpSvr7254.SetInputValue(1, 6) #short 기간선택 0:사용자지정, 1~3:개월 4:6개월 6: all
    # obj_CpSvr7254.SetInputValue(2, start_date) #long 시작일자
    # obj_CpSvr7254.SetInputValue(3, 20210205) #long 종료일자
    obj_CpSvr7254.SetInputValue(4, ord('0')) #char 매매비중 0:순매수 1:매매비중
    obj_CpSvr7254.SetInputValue(5, 0) #short 투자자
    obj_CpSvr7254.SetInputValue(6, ord('1')) #char 데이터 0: 순매수량 1: 금액
    
    start_flag = 0
    cnt = 0

    items = []
    
    while cnt < data_count:

        if start_flag:
            waitReqLimit(reqType.SISE)
        start_flag = 1

        obj_CpSvr7254.BlockRequest()
        count = obj_CpSvr7254.GetHeaderValue(1)

        # 통신 및 통신 에러 처리
        reqStatus = obj_CpSvr7254.GetDibStatus()
        reqRet = obj_CpSvr7254.GetDibMsg1()
        print("통신상태", reqStatus, reqRet)
        if reqStatus != 0:
            return False


        for i in range(count):
            item = {'코드': code}
            item['날짜'] = obj_CpSvr7254.GetDataValue(0, i)
            item['개인'] = obj_CpSvr7254.GetDataValue(1, i)
            item['외국인'] = obj_CpSvr7254.GetDataValue(2, i)
            item['기관계'] = obj_CpSvr7254.GetDataValue(3, i)
            item['금융투자'] = obj_CpSvr7254.GetDataValue(4, i)
            item['투신'] = obj_CpSvr7254.GetDataValue(5, i)
            item['은행'] = obj_CpSvr7254.GetDataValue(6, i)
            item['기타금융'] = obj_CpSvr7254.GetDataValue(7, i)

            items.append(item)

            cnt += 1

            if cnt % 50 == 0:
                print('CpSvr7254', code, cnt)


            if cnt == data_count:
                break

        # for item in items:
        #     print(item)

    return items

def call_StockChart(code, data_count, start_date):

    obj_StockChart.SetInputValue(0, code)  # string 종목코드
    obj_StockChart.SetInputValue(1, ord('2'))  # char '1': 기간, '2': 갯수
    # obj_StockChart.SetInputValue(3, start_date) #ulong 요청시작일
    #objRq.SetInputValue(2, 20210206) #ulong 요청종료일
    obj_StockChart.SetInputValue(4, data_count) #ulong 요청갯수
    #0: 날짜
    #5: 종가
    obj_StockChart.SetInputValue(5, [0,2,5,8,16,17,20,21]) #long list 필드 배열 [날짜, 시가, 종가, 거래량, 외국인현보유수량, 외국인현보유비율,기관순매수, 기관누적순매수]

    start_flag = 0
    cnt = 0
    items = []

    while cnt < data_count:
        
        if start_flag:
            waitReqLimit(reqType.SISE)
        start_flag = 1

        obj_StockChart.BlockRequest()
        count = obj_StockChart.GetHeaderValue(3)
        
        # 통신 및 통신 에러 처리
        reqStatus = obj_StockChart.GetDibStatus()
        reqRet = obj_StockChart.GetDibMsg1()
        print("통신상태", reqStatus, reqRet)
        if reqStatus != 0:
            return False

        for i in range(count):
            item = {}
            item['날짜'] = obj_StockChart.GetDataValue(0, i)
            item['시가'] = obj_StockChart.GetDataValue(1, i)
            item['종가'] = obj_StockChart.GetDataValue(2, i)
            item['거래량'] = obj_StockChart.GetDataValue(3, i)
            item['외국인현보유수량'] = obj_StockChart.GetDataValue(4, i)
            item['외국인현보유비율(%)'] = obj_StockChart.GetDataValue(5, i)
            item['기관순매수'] = obj_StockChart.GetDataValue(6, i)
            item['기관누적순매수'] = obj_StockChart.GetDataValue(7, i)

            items.append(item)
            
            cnt += 1

            if cnt % 50 == 0:
                print('StockChart', code, cnt)

            if cnt == data_count:
                break
        
        # for item in items:
        #     print(item)
    
    return items

def main():
    
    # 오늘날짜 : yyyymmdd
    now = datetime.now()
    today_date = now.strftime("%Y%m%d")
    
    # 추출한 데이터 수
    extract_data_num = 2000

    to_excel_data = []

    # 2021. 02. 07 기준 국내 시가총액 1 ~ 20위
    code_list = ['A005930', # 삼성전자
                     'A000660', # SK하이닉스
                     'A051910', # LG화학
                     'A005935', # 삼성전자우
                     'A035420', # NAVER
                    #  'A207940', # 삼성바이오로직스
                     'A005380', # 현대차
                     'A006400', # 삼성SDI
                     'A068270', # 셀트리온
                     'A000270', # 기아차
                     'A035720', # 카카오
                     'A012330', # 현대모비스
                     'A096770', # SK이노베이션
                     'A051910', # LG전자
                     'A051900', # LG생활건강
                     'A028260', # 삼성물산
                     'A005490', # POSCO
                     'A034730', # SK
                     'A036570', # 엔씨소프트
                     'A017670' # SK텔레콤
                     ]
    
    for code in code_list:
        
        print('----- Start [%s] -----' % code)

        data_7254 = call_CpSvr7254(code, extract_data_num, today_date)
        data_stockChart = call_StockChart(code, extract_data_num, today_date)

        for _data_7254, _data_stockChart in zip(data_7254, data_stockChart):
            data = dict(_data_7254, **_data_stockChart)
            # print(data)
            to_excel_data.append(data)
        
        print('----- Finish [%s] -----' % code)

    df = pd.DataFrame(to_excel_data)
    print(df)

    df.to_excel('data_%s.xlsx' % today_date)
    

    return


if __name__ == '__main__':


    print("Project: Golden Goose")

    obj_CpSvr7254 = win32com.client.Dispatch("CpSysDib.CpSvr7254")
    obj_CpStatus = win32com.client.Dispatch("CpUtil.CpCybos")
    obj_StockChart = win32com.client.Dispatch("CpSysDib.StockChart")

    main()
