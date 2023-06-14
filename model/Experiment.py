from konlpy.tag import Twitter as OriginalTwitter
from konlpy.tag import Okt
from ckonlpy.tag import Twitter

def read_data(filename):
    with open(filename, 'r',encoding='UTF-8') as f:  # 데이터 f 함수를 이용하여 불러오기
        data = [line.split('\t') for line in f.read().splitlines()]
        # f 함수를 이용하여 먼저 줄별로 split한 데이터를 반복문을 통해서 띄어쓰기를 구분자로 split한 것을 리스트로 반환

        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]  ##맨위에 스키마는 제외
    return data


train_df = read_data('C:/Users/User/PycharmProjects/Crawling/train.txt')
print("훈련 데이터 개수는: "+str(len(train_df))+'개')
#이 과정 거치면 ['1', '수강취소 혹시 거절당할수도있어??', '1'], ~
test_df = read_data('C:/Users/User/PycharmProjects/Crawling/test.txt')
#print("테스트 데이터 개수는: "+str(len(test_df))+'개')

from konlpy.tag import Okt
# 밑의 tokenizing함수를 이해하기 편하도록 들어놓은 예시
# tokenizing함수의 반복문 구조를 보시면 아래에 나온 결과를 반복문으로 '/'를 구분자로 결합시킵니다.
okt=Okt(max_heap_size=1024*7)

# 하나의 문장을 토큰화 한 후 텍스트와 품사태깅을 / 구분자로 묶어준다.
def tokenizing(docs):
    return ['/'.join(t) for t in okt.pos(docs, norm=True, stem=True)]


train_pos = []  # 훈련데이터
#
# with open('C:/Users/LeeSangJae/Desktop/learning/train_pos.txt','r',encoding='UTF-8') as file_point:
#     for s in file_point:
#         train_pos.append(s[1:-2].replace('\'',"").split(", "))
#         #list.append(s.split(','))
test_pos = []  # 테스트 데이터

for row in train_df:
    try:
        train_pos0 = [tokenizing((row[2])), row[3]]  ## row2이 두번째 항목 (text), row3가 라벨
        # 리스트 안에 한문장에 대해서 위에서 만든 tokenizing함수를 통해서 [[토큰화텍스트],긍/부정 여부]를
        # 리스트의 각문장별로 요소로 넣는다.
        train_pos.append(train_pos0)
        """
        #new_tokenizing=[]
        #for w in tokenizing(row[2]):
        #    if w not in stop_words:
        #        new_tokenizing.append(w)
        #train_pos0 = [new_tokenizing, row[3]]  ## row2이 두번째 항목 (text), row3가 라벨
        """
        #토크나이징 (row[2])자리에 new_t가 들어가야 함
        # 리스트 안에 한문장에 대해서 위에서 만든 tokenizing함수를 통해서 [[토큰화텍스트],긍/부정 여부]를
        # 리스트의 각문장별로 요소로 넣는다.

    except:
        pass

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
print("중복을 제외한 토큰 개수는: "+str(len(set(text.tokens)))+'개')
most_words = open('C:/Users/User/PycharmProjects/Crawling/mostwords.txt','w',encoding='utf-8')
for word in text.vocab().most_common(5000):
    most_words.write(str(word)+'\n')
most_words.close()

selected_words = [f[0] for f in text.vocab().most_common(5000)]  #n개
new_selected_words=selected_words
#불용어제거

stop_words = list()
with open('C:/Users/User/PycharmProjects/Crawling/mostwords.txt','r',encoding='UTF-8') as file_point:
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

def term_frequency(doc):
    return [doc.count(word) for word in new_selected_words]


train_x = [term_frequency(d) for d, _ in train_pos]
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
from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint
#모델 구성하기
# tensorflow.keras를 활용하여 모델의 층 입력하기  => 여기를 이해해야 함.
r = [0.99, 0.98, 0.97, 0.96, 0.95]

for sz in range(300, 10000, 1000):
    for param in range(100, 1000, 100):
        for i in r:
            model = models.Sequential()
            # model.add(layers.Dense(100, activation='relu', input_shape=(len(new_selected_words),)))  # 10000개를 추출했으므로 shape는10000 => n개 , 벡터를 위에서 최빈도수 5000개로 잡았으니까 쉡이프가 5천?
            model.add(layers.Dense(param, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))

            """
            # 모델 불러오기   ############################## 불러올땐 이거 사용
            from keras.models import load_model
            model = load_model('every_time_model')
            """
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
            mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
            # 모델 생성 (최적화) / 모델 엮기
            model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                          loss=losses.binary_crossentropy,
                          metrics=[metrics.binary_accuracy])

            class_weight = {1.0:1-i,
                            0.0:i}  #925베스트
            # 모델 학습
            model.fit(x_train, y_train, epochs=5, batch_size=sz,validation_data=(x_test,y_test),class_weight=class_weight)


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

                score = float(model.predict(data)) #새로운 데이터를 받으면 결과 예측`

                if(score > 0.5):
                    return ("[{}]는 {:.2f}% 학교에 대한 불만이 아닙니다.\n".format(text, score * 100)+'\n')
                else:
                    return ("[{}]는 {:.2f}% 학교에 대한 불만입니다.\n".format(text, (1 - score) * 100)+'\n')


            temp = open('C:/Users/User/PycharmProjects/Crawling/test.txt','r',encoding='utf-8')  #테스트하고싶은 데이터
            model_result = open('C:/Users/User/PycharmProjects/Crawling/model_evaluate/정확도{}%_다층퍼셉_가중치{}_배치사이즈{}_파라미터{}.txt'.format(round(results[1] , 2)*100, i, sz, param),'w',encoding='utf-8') #테스트 결과를 출력할 파일
            while True:
                sentence = temp.readline()
                if not sentence:
                    break
                mylist = sentence.split('\t')
                model_result.write(predict_pos_text(mylist[2]))

temp.close()
model_result.close()