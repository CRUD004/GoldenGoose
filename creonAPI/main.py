import sys
import win32com.client
import os


objRq = win32com.client.Dispatch("CpSysDib.CpSvr7254")
objRq.SetInputValue(0, 'A000270') #string 종목코드
objRq.SetInputValue(1, 0) #short 기간선택 0:사용자지정, 1~3:개월 4:6개월
objRq.SetInputValue(2, 20200301) #long 시작일자
objRq.SetInputValue(3, 20210205) #long 종료일자
objRq.SetInputValue(4, ord('0')) #char 매매비중 0:순매수 1:매매비중
objRq.SetInputValue(5, 0) #short 투자자
objRq.SetInputValue(6, ord('1')) #char 데이터 0: 순매수량 1: 금액

while True:
    objRq.BlockRequest()
    count = objRq.GetHeaderValue(1)

    if count == 0:
        print("no more data")
        break
    
    print("종목코드: ", objRq.GetHeaderValue(0))
    print("총 데이터: ", count)
    print("시작일자: ", objRq.GetHeaderValue(2))
    print("종료일자: ", objRq.GetHeaderValue(3))

    for i in range(count):
        print("date:{} 개인: {} 기관: {} 외국인: {}".format(objRq.GetDataValue(0,i), objRq.GetDataValue(1,i), objRq.GetDataValue(2,i), objRq.GetDataValue(3,i)))

''' 
#종목의 일자별 정보를 가져올 수 있음
objRq = win32com.client.Dispatch("CpSysDib.StockChart")

objRq.SetInputValue(0, 'A000270')  # string 종목코드
objRq.SetInputValue(1, ord('1'))  # char '1': 기간, '2': 갯수
objRq.SetInputValue(3, 20210101) #ulong 요청시작일
#objRq.SetInputValue(2, 20210206) #ulong 요청종료일
objRq.SetInputValue(4, 10) #ulong 요청갯수

#0: 날짜
#5: 종가
objRq.SetInputValue(5, [0,5,20,21]) #long list 필드 배열

objRq.BlockRequest()

print("종목코드: ", objRq.GetHeaderValue(0))
print("필드: ", objRq.GetHeaderValue(2))
cnt = objRq.GetHeaderValue(3)
print("수신개수: ", cnt)

for i in range(cnt-1, 0,-1):
    #print("date:{} 종가: {} ".format(objRq.GetDataValue(0,i), objRq.GetDataValue(1,i)))
    print("date:{} 종가: {} 기관순매수: {} 기관누적순매수: {}".format(objRq.GetDataValue(0,i), objRq.GetDataValue(1,i), objRq.GetDataValue(2,i), objRq.GetDataValue(3,i)))
'''
