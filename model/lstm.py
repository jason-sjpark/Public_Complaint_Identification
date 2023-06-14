import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
import numpy as np
from konlpy.tag import Twitter
from konlpy.tag import Okt
from ckonlpy.tag import Twitter

train_data = pd.read_table('C:/Users/GomZoo/PycharmProjects/제대로/finaltrain.txt') ##아무거나
test_data = pd.read_table('C:/Users/GomZoo/PycharmProjects/제대로/test_5월8천.txt') ##테스트5월

train_data['text'].nunique(), train_data['label'].nunique() ##중복확인
train_data.drop_duplicates(subset=['text'], inplace=True) ## text열에서 중복있으면 제거

train_data['label'].value_counts().plot(kind = 'bar')
#plt.show()   ##라벨 비율 보기
#훈련데이터 정제
print(train_data.groupby('label').size().reset_index(name = 'count')) ## 라벨 개수 세기
train_data['text'] = train_data['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]","") ##정규표현식으로 한글과 공백을 제외하고 모두 제거, 숫자는 살릴 필요 있지 않을까...? 일단 다 지워p
#[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9 ]
train_data = train_data.dropna(how = 'any') #정규표현식 쓰고 null인 애들 제거
#테스트데이터 정제
test_data['text'].nunique(), train_data['label'].nunique()
test_data.drop_duplicates(subset = ['text'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['text'] = test_data['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]","") # 정규 표현식 수행
test_data['text'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거

#토큰화

stop_words = list()
with open('C:/Users/GomZoo/PycharmProjects/제대로/mostwords.txt','r',encoding='UTF-8') as file_point:
    for s in file_point:
        if "Josa" in s:
            l = s.split("'")
            stop_words.append(l[1][:l[1].find('/')])
        if "Punctuation" in s:
            l = s.split("'")
            stop_words.append(l[1][:l[1].find('/')])
        if "Suffix" in s:
            l = s.split("'")
            stop_words.append(l[1][:l[1].find('/')])
        if "Number" in s:
            l = s.split("'")
            stop_words.append(l[1][:l[1].find('/')])
        if "Exclamation" in s:
            l = s.split("'")
            stop_words.append(l[1][:l[1].find('/')])

#twitter = Twitter()
okt=Okt(max_heap_size=1024*7)
"""
add_words = list()
with open('C:/Users/LeeSangJae/Desktop/learning/추가단어.txt','r',encoding='UTF-8') as file_point:
    for s in file_point:
        add_words.append(s[:-1])
twitter.add_dictionary(add_words,'Noun')
"""
X_train = [] ##훈련데이터
"""
for sentence in train_data['text']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stop_words] # 불용어 제거
    X_train.append(temp_X)
"""
with open('C:/Users/GomZoo/PycharmProjects/제대로/lstm트레인전처리불용어제거.txt','r',encoding='UTF-8') as file_point:
    for s in file_point:
        X_train.append(s[1:-2].replace('\'',"").split(", "))
print(len(X_train))
X_test = []  ##테스트데이터
for sentence in test_data['text']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stop_words] # 불용어 제거
    X_test.append(temp_X)
# with open('C:/Users/LeeSangJae/Desktop/learning/lstm테스트전처리.txt','r',encoding='UTF-8') as file_point:
#     for s in file_point:
#         X_test.append(s[1:-2].replace('\'',"").split(", "))

##정수 인코딩
tokenizer = Tokenizer()
print("영번째\n")
tokenizer.fit_on_texts(X_train)
print("첫번째\n")


threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
vocab_size = total_cnt - rare_cnt + 1 # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
print('단어 집합의 크기 :',vocab_size)
# 훈련,테스트 둘 다 정수 인코딩
tokenizer = Tokenizer(vocab_size)  #전체 단어를 다 쓴다는 의미인 듯
tokenizer.fit_on_texts(X_train)
print("토큰에 보캡적용후\n")
#print(tokenizer.word_index)
#x는 텍스트
X_train = tokenizer.texts_to_sequences(X_train)  #여기서 X_train = ['안녕', '하다', '오늘', '까지', 'asb', '124']
X_test = tokenizer.texts_to_sequences(X_test)

print("길이:"+str(len(X_test)))
#y는 라벨
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])
print("길이:"+str(len(y_test)))
#빈샘플제거
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
drop_test = [index for index, sentence in enumerate(X_test) if len(sentence) < 1]
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
X_test = np.delete(X_test, drop_test, axis=0)
y_test = np.delete(y_test, drop_test, axis=0)
#패딩(서로 다른 길이의 샘플들의 길이를 동일하게 맞춰주는 거)
print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
#plt.show()

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 80 #리뷰 최대 길이
below_threshold_len(max_len, X_train) #맥스렌 이하인 샘플의 비율이 얼마인지
#모든 샘플의 길이를 맥스렌으로 맞춤
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
print("두번째\n")
#print(tokenizer.word_index)
print(X_test[:2])
print(vocab_size)
print(len(X_train))

#학습모델만들기
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.recurrent import RNN,SimpleRNN
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def predict_pos_text(text,r,j,k,q):  #가중치 1층 2층
    table = pd.read_table('C:/Users/GomZoo/PycharmProjects/제대로/'+text+'.txt')
    #table['text'] = table['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]","")
    XX_train=[]
    #drop=[]
    # for sentence in table['text']:
    #     temp_X = []
    #
    #     temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    #     temp_X = [word for word in temp_X if not word in stop_words]  # 불용어 제거
    #     X_train.append(temp_X)
    with open('C:/Users/GomZoo/PycharmProjects/제대로/lstm테스트전처리불용어제거.txt', 'r', encoding='UTF-8') as file_point:
        for s in file_point:
            XX_train.append(s[1:-2].replace('\'', "").split(", "))

    print("세번째\n")
    print(tokenizer.word_index)
    print(vocab_size)
    XX_train = tokenizer.texts_to_sequences(XX_train)

    #X_train = np.delete(X_train, drop_train, axis=0)
    XX_train = pad_sequences(XX_train, maxlen=max_len)
    XX_train = np.array(XX_train)


    model_result = open('C:/Users/GomZoo/PycharmProjects/제대로/lstm/최종테스트'+"가중치"+str(r)+" 1층"+str(j)+" 2층"+str(k)+" 정확도"+str(q)+'.txt', 'w', encoding='utf-8')

    i = 0
    for data in XX_train:
        a=[]
        a.append(data)
        a=np.asarray(a)
        if i==0: print(a)
        score = float(loaded_model.predict(a)) #새로운 데이터를 받으면 결과 예
        if(score > 0.4):
            model_result.write("[{}]는 {:.2f}% 학교에 대한 불만이 아닙니다.\n".format(table['text'][i], score * 100)+'\n')
        else:
            model_result.write("[{}]는 {:.2f}% 학교에 대한 불만입니다.\n".format(table['text'][i], (1 - score) * 100)+'\n')
        i=i+1

rrrrr= open('C:/Users/GomZoo/PycharmProjects/제대로/lstm/결과.txt', 'a', encoding='utf-8')
r1=[0.99]
rr=[]
for i in r1:
    for j in [128,256,512,1024]:
        for k in [64,128,256,512]:
            model = Sequential()
            model.add(Embedding(vocab_size, j))
            model.add(LSTM(k))
            model.add(Dense(1, activation='sigmoid'))

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
            mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
            class_weight = {1.0:1-i,
                            0.0:i}
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
            history = model.fit(X_train, y_train, epochs=7, callbacks=[es, mc], batch_size=1000, validation_split=0.1,class_weight=class_weight)
            loaded_model = load_model('best_model.h5')
            print("모델: ")
            print(loaded_model)
            rr.append(float(loaded_model.evaluate(X_test, y_test)[1]))
            print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1])+'=> i='+str(i))
            q = (loaded_model.evaluate(X_test, y_test)[1])
            rrrrr.write(str(i) + " " + str(j) + " " + str(k) + " " + str(q) + " 테스트 정확도: %.4f" % q + '=> i=' + str(i) + "\n")
            predict_pos_text("test_5월8천",i,j,k,q)
            rrrrr.write(str(i)+" "+str(j)+" "+str(k)+" "+str(q)+" 테스트 정확도: %.4f" % q+'=> i='+str(i)+"\n")
rrrrr.close()
print("실험결과")
print(rr)
## 요 모델로 평가해보