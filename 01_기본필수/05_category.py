"""05 기본필수: qa와 summary를 서로 다른 0/1 열로 표현합니다."""
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoding 과정
# "qa", "summary"와 같은 문자열 데이터 -> 0,1 숫자형태로 변환.
train = [["qa"], ["summary"], ["qa"]]
valid = [["summary"], ["qa"]]

# sparse_output=False는 결과를 바로 볼 수 있는 숫자 표로 받는 설정입니다.
encoder = OneHotEncoder(sparse_output=False)
# train 훈련 데이터로 어떤 카테고리가 있는지 학습 = fit 과정
# 0번 자리 = qa, 1번 자리 = summary 로 기준 지정.
encoder.fit(train)

# TODO B05-1: encoder.transform(valid).tolist()로 실제 변환 결과를 구하세요.
# valid 검증 데이터 -> 숫자표로 변환(transform)
# fit을 한 다음에 transform 해야됨
valid_ready = encoder.transform(valid).tolist()

# 컴퓨터가 세운 기준 = ["qa", "summary"] 순서
# "qa" = [1,0] / "summary" = [0,1]
print("열 순서:", encoder.categories_[0].tolist())
print("valid 변환:", valid_ready)
