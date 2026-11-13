import pandas as pd, re, sys
src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, encoding='utf-8-sig')
d['labeler'] = d['labeler'].astype(str)
# place: (relevance, risk, nearest, reason) ; None = keep
F = {
 # 1. 공식 위치 아님 → 0.9 미만으로 구간 분배
 '하역장':(0.85,None,'inbound','목록에 없는 장소로, 차량에서 짐을 내리는 곳이라 입고장과 기능이 가장 가깝습니다.'),
 '하차장':(0.85,None,'inbound','목록에 없는 장소로, 차량에서 짐을 내리는 곳이라 입고장과 기능이 가장 가깝습니다.'),
 '상차장':(0.85,None,'outbound','목록에 없는 장소로, 차량에 짐을 싣는 곳이라 출고장과 기능이 가장 가깝습니다.'),
 '적재장':(0.8,None,'outbound','목록에 없는 장소로, 짐을 쌓아 싣는 곳이라 출고장과 가깝지만 이름만으로는 확실하지 않습니다.'),
 '입고 대기 구역':(0.85,None,'inbound','목록에 없는 장소로, 입고장 바로 앞 대기 공간이라 입고장과 가깝습니다.'),
 '출하 대기 구역':(0.85,None,'outbound','목록에 없는 장소로, 출고장 바로 앞 대기 공간이라 출고장과 가깝습니다.'),
 '도크':(0.8,None,'none','목록에 없는 장소로, 입고와 출고 양쪽에 쓰여 어느 공식 위치인지 정할 수 없습니다.'),
 '보관 랙':(0.8,None,'none','목록에 없는 보관 장소로, 선반과 비슷하지만 어느 칸인지 알 수 없습니다.'),
 '재고 구역':(0.8,None,'none','목록에 없는 보관 장소로, 선반 구역과 비슷하지만 어느 칸인지 알 수 없습니다.'),
 '크로스도킹 구역':(0.75,None,'none','목록에 없는 물류 장소로, 입고와 출고를 함께 다뤄 어느 공식 위치인지 정할 수 없습니다.'),
 '피킹 구역':(0.75,None,'none','목록에 없는 물류 장소로, 선반에서 물건을 고르는 작업 공간이며 포장대와는 다른 곳입니다.'),
 '검수대':(0.75,None,'inbound',None),
 '분류 라인':(0.75,None,'outbound',None),
 '반품 구역':(0.7,None,'inbound',None),
 '반송 구역':(0.7,None,'inbound',None),
 '팔레트 보관소':(0.7,None,'none',None),
 '임시 보관소':(0.7,None,'none',None),
 '라벨 작업대':(0.7,None,'packing',None),
 '랩핑기 앞':(0.7,None,'packing',None),
 '컨베이어 벨트':(0.7,None,'none',None),
 '대형 화물 구역':(0.7,None,'none',None),
 '소형 화물 구역':(0.7,None,'none',None),
 '냉장 보관소':(0.7,None,'none',None),
 '냉동 창고':(0.7,None,'none',None),
 '공박스 보관소':(0.65,None,'packing',None),
 '지게차 충전소':(0.6,None,'none','목록에 없는 장소로, 지게차용 충전 설비라 로봇 충전소로 보면 안 되며 장비 충돌 위험이 있습니다.'),
 '소모품 창고':(0.6,None,'none',None),
 '불량품 구역':(0.6,None,'none',None),
 '계량대':(0.6,None,'inbound',None),
 '스캔대':(0.6,None,'inbound',None),
 '자재실':(0.5,None,'none','창고 안에 있지만 물류 작업 구역은 아니며 공식 위치와 대응되지 않습니다.'),
 'E구역 2번 선반':(0.6,None,'none','선반이라는 점은 물류와 관련 있지만 E구역은 목록에 없는 구역이라 대응되는 공식 위치가 없습니다.'),
 'H3':(0.6,None,'none','선반 번호 형식이지만 H구역은 목록에 없는 구역이라 대응되는 공식 위치가 없습니다.'),
 '화물 엘리베이터':(0.6,None,'none',None),
 '지게차 통로':(0.6,None,'none',None),
 '하역 도크 끝':(0.7,None,'inbound',None),
 '3층 선반':(0.6,None,'none',None),
 # 2. 사무·생활 공간 위험률 기준 통일 (사람이 있는 실내 공간 0.4)
 '사무실':(None,0.4,None,None),'회의실':(None,0.4,None,None),'탈의실':(None,0.4,None,None),
 '휴게실':(None,0.4,None,None),'식당':(None,0.4,None,None),'화장실':(None,0.4,None,None),
 '로비':(None,0.4,'none',None),'안내 데스크':(None,0.4,None,None),'경비실':(None,0.4,None,None),
 '관리실':(None,0.4,None,None),'의무실':(None,0.4,None,None),'흡연실':(None,0.4,None,None),
 '수유실':(None,0.4,None,None),'탕비실':(None,0.4,None,None),'센터장실':(None,0.4,None,None),
 '교육장':(None,0.4,None,None),
 '샤워실':(None,0.5,None,'물류 작업과 무관하며, 바닥에 물이 있어 미끄러짐이나 로봇 고장 위험이 있습니다.'),
 '2층 사무실':(None,0.8,None,'물류 작업과 무관하며, 다른 층으로 가려면 계단이나 엘리베이터를 거쳐야 해 추락 위험이 있습니다.'),
 '서버실':(0.1,None,'none','물류 작업과 무관하며, 전기·전산 설비가 많아 충돌 시 장비 피해가 큽니다.'),
 # 3. 위험한데 낮게 나온 곳
 '주차장':(None,0.8,None,'건물 밖 공간으로 물류 작업과 무관하며, 차량이 다녀 충돌 위험이 큽니다.'),
 '정문':(None,0.6,'none','건물 출입구로 물류와 간접적으로만 관련되며, 건물 밖으로 나가게 되고 사람·차량과 마주칠 위험이 있습니다.'),
 '지하 창고':(None,0.7,None,'목록에 없는 다른 층의 보관 공간으로, 가려면 계단이나 경사로를 거쳐야 해 추락 위험이 있습니다.'),
 # 4. nearest
 'B열 2번 칸':(1.0,None,'B2','선반 B2 를 부르는 표기로, make_data.py 의 선반 표기 6가지 중 하나라 해석 모델이 B2 로 학습한 공식 표기입니다.'),
 '전기실':(None,None,'none',None),'기계실':(0.1,None,'none',None),
 '배터리 보관실':(0.1,None,'none','로봇 충전소와는 다른 곳이며, 인화성 물질이 있어 접근 시 화재 위험이 큽니다.'),
 '화학약품 보관실':(0.1,None,None,None),'위험물 보관소':(0.1,None,None,None),
 '출입구':(None,None,'none',None),'후문':(None,None,'none',None),'경사로':(None,None,'none',None),
}
miss=[p for p in F if p not in set(d.place)]
assert not miss, miss
for p,(rel,risk,near,reason) in F.items():
    i=d.index[d.place==p][0]
    if rel is not None: d.at[i,'relevance']=rel
    if risk is not None: d.at[i,'risk']=risk
    if near is not None: d.at[i,'nearest']=near
    if reason is not None: d.at[i,'reason']=reason
    d.at[i,'labeler']='qwen3.5:9b+검토'
# 2. 창고 밖 위험률 통일
m=d.group=='outside'
d.loc[m,'risk']=0.3; d.loc[m,'nearest']='none'; d.loc[m,'labeler']='qwen3.5:9b+검토'
d.loc[m,'reason']=d.loc[m,'place']+'은(는) 창고 밖 장소로 물류 작업과 무관합니다. 로봇이 실제로 갈 수 없어 잘못 실행해도 창고 안 엉뚱한 곳으로 가는 정도의 피해로 봅니다.'
# 5. 러시아어 섞임
r=d.reason.str.contains('[а-яА-Я]')
d.loc[r,'reason']=d.loc[r,'reason'].str.replace('туда로','그곳으로').str.replace('туда','그곳으로')
assert not d.reason.str.contains('[а-яА-Я]').any()
d.to_csv(out,index=False,encoding='utf-8-sig')
print(d.groupby('group')[['relevance','risk']].agg(['mean','min','max']).round(2))
print('>=0.9 non-official:', d[(d.relevance>=0.9)&(d.group!='official')].place.tolist())
print('edited rows:', (d.labeler!='qwen3.5:9b').sum())
