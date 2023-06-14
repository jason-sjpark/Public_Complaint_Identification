from pykospacing import spacing
from konlpy.tag import Twitter as OriginalTwitter
from konlpy.tag import Okt
from ckonlpy.tag import Twitter
from pykospacing import spacing

def read_data(filename):
    with open(filename, 'r',encoding='UTF-8') as f:  # 데이터 f 함수를 이용하여 불러오기
        data = [line.split('\t') for line in f.read().splitlines()]
        # f 함수를 이용하여 먼저 줄별로 split한 데이터를 반복문을 통해서 띄어쓰기를 구분자로 split한 것을 리스트로 반환

        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]  ##맨위에 스키마는 제외
    return data


train_df = read_data('최종트레인셋.txt')
print("훈련 데이터 개수는: "+str(len(train_df))+'개')
#이 과정 거치면 ['1', '수강취소 혹시 거절당할수도있어??', '1'], ~
test_df = read_data('5월테스트.txt')
#print("테스트 데이터 개수는: "+str(len(test_df))+'개')

from konlpy.tag import Okt
# 밑의 tokenizing함수를 이해하기 편하도록 들어놓은 예시
# tokenizing함수의 반복문 구조를 보시면 아래에 나온 결과를 반복문으로 '/'를 구분자로 결합시킵니다.
twitter = Twitter()

add_words = list()
with open('추가단어.txt','r',encoding='UTF-8') as file_point:
    for s in file_point:
        add_words.append(s[:-1])
twitter.add_dictionary(add_words,'Noun')

# 하나의 문장을 토큰화 한 후 텍스트와 품사태깅을 / 구분자로 묶어준다.
def tokenizing(docs):
    return ['/'.join(t) for t in twitter.pos(docs, norm=True, stem=True)]


train_pos = []  # 훈련데이터
test_pos = []  # 테스트 데이터
for row in train_df:
    try:
        train_pos0 = [tokenizing((row[2])), row[3]]  ## row2이 두번째 항목 (text), row3가 라벨
        # 리스트 안에 한문장에 대해서 위에서 만든 tokenizing함수를 통해서 [[토큰화텍스트],긍/부정 여부]를
        # 리스트의 각문장별로 요소로 넣는다.
        train_pos.append(train_pos0)
        """
        new_tokenizing=[]
        for w in tokenizing(row[2]):
            if w not in stop_words:
                new_tokenizing.append(w)
        train_pos0 = [new_tokenizing, row[3]]  ## row2이 두번째 항목 (text), row3가 라벨
        """
        #토크나이징 (row[2])자리에 new_t가 들어가야 함
        # 리스트 안에 한문장에 대해서 위에서 만든 tokenizing함수를 통해서 [[토큰화텍스트],긍/부정 여부]를
        # 리스트의 각문장별로 요소로 넣는다.

    except:
        pass
#이과정 거치면
#train_pos=[[['수강/Noun', '취소/Noun', '혹시/Noun', '거절/Noun', '당하다/Adjective', '있다/Adjective', '??/Punctuation'], '1'],
for row in test_df:
    try:
        test_pos0 = [tokenizing((row[2])), row[3]]
        test_pos.append(test_pos0)
    except:
        pass

#위에서 만든 데아터에서 긍/부정을 제외하고 token에 넣어준다. [[a],b] 에서 a만 넣는다고 생각하면 됨
tokens = [t for d in train_pos for t in d[0]]
print("총 토큰의 개수는: "+str(len(tokens))+"개") ## 총 몇개의 토큰이 있는지
#tokens에 텍스트만 들어가는 거네

##요코드로 부정 긁어낸다음에 키워드 추출해도 될 듯
import nltk
text = nltk.Text(tokens,name='NMSC')#nltk라이브러리를 통해서 텍스트 데이터 나열
len(set(text.tokens))#35425개의 고유 텍스트가 존재
print("중복을 제외한 토큰 개수는: "+str(len(set(text.tokens)))+'개') ## 중복 제외하고 토큰 몇개인지
##set(text.tokens) = = {'ㅁㅊ/KoreanParticle', '호/Noun', '이라도/Foreign', '품타/Noun', '족발/Noun', '연결하다/Adjective', '노스/Noun', '기숙사/Noun', '수학/Noun', '갈아타다/Verb', '싶다/Verb',
#text.vocab().most_common(200) #vocab().most_common(10) - 텍스트 빈도 상위 10개 보여주기 즉, count_values()를 통해서 내림차순한 것과 같습니다.
most_words = open('최종셋많이나온단어.txt','w',encoding='utf-8')
for word in text.vocab().most_common(13000):
    most_words.write(str(word)+'\n')
most_words.close()

#나열하니까 이렇게 나오는데... 조사를 좀 없애야 될듯[('?/Punctuation', 21), ('하다/Verb', 12), ('가/Josa', 9), ('있다/Adjective', 7), ('에/Josa', 6), ('없다/Adjective', 6), ('생각/Noun', 6), ('되다/Verb', 6), ('이/Josa', 6), ('도/Josa', 6)]


#단어 빈도수가 높은 n개의 단어만 사용하여 각 리뷰 문장마다의 평가지표로 삼는다.
selected_words = [f[0] for f in text.vocab().most_common(13000)]  #n개
new_selected_words=selected_words
#불용어제거

stop_words = list()
with open('최종셋많이나온단어.txt','r',encoding='UTF-8') as file_point:
    for s in file_point:
        if "Josa" in s:
            l = s.split("'")
            stop_words.append("'" + l[1] + "'")
        if "Punctuation" in s:
            l = s.split("'")
            stop_words.append("'" + l[1] + "'")
        if "Suffix" in s:
            l = s.split("'")
            stop_words.append("'" + l[1] + "'")
        if "Number" in s:
            l = s.split("'")
            stop_words.append("'" + l[1] + "'")
        if "Exclamation" in s:
            l = s.split("'")
            stop_words.append("'" + l[1] + "'")
        if "KoreanParticle" in s:
            l = s.split("'")
            stop_words.append("'" + l[1] + "'")

new_selected_words=[]
for w in selected_words :
    if w not in stop_words:
        new_selected_words.append(w)

#위에서 상위 n개 나열한 거 빈도수 빼고 보여줌
#print(selected_words)

#많이 나온 단어 파일로 만들기
"""
most_words = open('C:/Users/LeeSangJae/Desktop/learning/train에서 많이 나온 단어.txt','w',encoding='utf-8')

for word in selected_words:
    most_words.write(word)
most_words.close()
"""
#term_frequency()함수는 위에서 만든 selected_words의 갯수에 따라서 각 리뷰와 매칭하여 상위 텍스트가
#각 리뷰에 얼만큼 표현되는지 빈도를 만들기 위한 함수
def term_frequency(doc):
    return [doc.count(word) for word in new_selected_words]


train_x = [term_frequency(d) for d, _ in train_pos]
#train_x만 설명하자면 위의 결과로 도출되는 train_x의 구조는 아래와 같다
#[[1번째리뷰를 상위 10000개와 각각 매칭하여 각 10000개의 단어가 해당 문장에 얼마나 포함되는지를 확인]]
#리스트 차원으로 표현하면 [[10000개],[10000개],[10000개]....[10000개]] 가장 바깥 리스트의 갯수는 기존 train_pos의 리뷰 갯수와 같다.
test_x = [term_frequency(d) for d, _ in test_pos]
train_y = [c for _, c in train_pos] #train_pos데이터에서 각 리뷰별 긍/부정 여부 데이터이므로 train_pos의 리뷰갯수와 같은 사이즈이다.
test_y = [c for _, c in test_pos]

#모델링을 하기 위해 리스트로 되어 있는 데이터 형식을 array로 바꿔주고 dtype도 실수로 바꿔준다.
import numpy as np

x_train = np.asarray(train_x).astype('float32')

x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')
#요까지 데이터셋 준비
####################################################
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

#모델 구성하기
# tensorflow.keras를 활용하여 모델의 층 입력하기  => 여기를 이해해야 함.
model = models.Sequential()
#은닉층은 렐루가 성능이 젤 좋단다
model.add(layers.Dense(3000, activation='relu', input_shape=(len(new_selected_words),)))  # 10000개를 추출했으므로 shape는10000 => n개 , 벡터를 위에서 최빈도수 5000개로 잡았으니까 쉡이프가 5천?
#64는 바꿔가면서 모델 성능 바꿀 수 있음
#input_shape => 5000개를 특성으로 비교하겠다! (=input_dim)
#(입력 뉴런수가 5000개)
model.add(layers.Dense(2000, activation='relu'))
#Dense 레이어는 보통 출력층 이전의 은닉층으로도 많이 쓰이고, 영상이 아닌 수치자료 입력 시에는 입력층으로도 많이 쓰입니다. 이 때 활성화 함수로 ‘relu’가 주로 사용됩니다.
#‘relu’는 학습과정에서 역전파 시에 좋은 성능이 나는 것으로 알려져 있습니다.
#가장 자주 사용되는 활성화함수

#출력층
model.add(layers.Dense(1, activation='sigmoid'))  # 이진 분류 문제 이므로 0과 1을 나타내는 출력 뉴런 하나만 있으며 됨, 활성화 함수는 sigmoid
#Dense(8, input_dim=4, init='uniform',activation='relu')
#8개 출력, input_dim=> 입력 뉴런 수, init 생략하면 자동적으로 uniform,
#이진 분류할거라서 sigmoid가 좋음
#s
"""
# 모델 불러오기   ############################## 불러올땐 이거 사용
from keras.models import load_model
model = load_model('every_time_model')
"""

# 모델 생성 (최적화) / 모델 엮기
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

#이진 분류라서 binary_crossentropy(오브젝트펑션)
#옵티마이저는 선택
#메트릭

class_weight = {1.0:0.02,
                0.0:0.98}  #925베스트
# 모델 학습
model.fit(x_train, y_train, epochs=5, batch_size=300,validation_data=(x_test,y_test),class_weight=class_weight)  #ex 샘플돌릴때 512개씩 잘라서 돌림, 열번 반복, 에포크마다 테스트 데이터로 평가
#val_loss 가 계속 줄어드는지 확인!! (가장 낮은 시점의 에포크를 찾아서 써야 함!)
#트레인 데이터 수만큼 10번 반복 총 트레인 * 10만큼 이 학습하는 샘플 수
#트레인 * 10 / 512 만큼 네트워크 갱신
#fit 할때 문제와 정답을 줘야 함
#batch size => 몇 문항 풀고 업데이트 할거냐
#업데이트를 한다, 네트워크를 갱신한다 => 옵티마이저
#배치사이즈에따라 메모리용량, 학습속도 차이남
#에포크 => 반복횟수 ex)100문제를 한번 다 푸는게 에포크 1 열번 풀면 에포크 10 같은 문제집을 푼다고 생각(같은 문제집이라도 반복해서 풀면 학습이 일어남)


#모델 사용하기
results = model.evaluate(x_test, y_test)
#모델 저장
#model.save('every_time_model')
# 예측 결과
print(results)  # 83%의 정확도를 가진다.

##모델로 채점해보는 코드

def predict_pos_text(text):
    token = tokenizing(text) #okt.pos로 토큰화한 단어를 정리
    tf =term_frequency(token)#토큰화된 단어를 이용해서 가장 많이 등장하는 단어와의 빈도수 체크

    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    #data는 그럼 입력받는 text가 한 문장만 받기 때문에 가장 바깥 리스트의 요소 갯수는 1이 될 것이고
    #마찬가지로 리스트 안 리스트 요소내의 갯수는 10000개가 될 것이다.
    #np.expand_dims??
    score = float(model.predict(data)) #새로운 데이터를 받으면 결과 예측
    if(score > 0.5):
        return ("[{}]는 {:.2f}% 학교에 대한 불만이 아닙니다.\n".format(text, score * 100)+'\n')
    else:
        return ("[{}]는 {:.2f}% 학교에 대한 불만입니다.\n".format(text, (1 - score) * 100)+'\n')


temp = open('5월부정.txt','r',encoding='utf-8')  #테스트하고싶은 데이터
model_result = open('최종평가.txt','w',encoding='utf-8') #테스트 결과를 출력할 파일
while True:
    sentence = temp.readline()
    if not sentence:
        break
    mylist = sentence.split('\t')
    model_result.write(predict_pos_text(mylist[2]))

temp.close()
model_result.close()