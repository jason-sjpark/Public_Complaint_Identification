from selenium import webdriver
from bs4 import BeautifulSoup
from collections import defaultdict
import time
import datetime
############################
def encoding(json_data):
    json_data = json_data.encode('utf-8')
    json_data = json_data.decode('unicode_escape')
    return json_data

def totxttime(json_data, text_num):
    txtfile = open('2019_6.txt', 'a', -1, "utf-8")
    txtfile.write(str(text_num) + "\t")
    txtfile.write(json_data)
    txtfile.write("\t")
    #print(text_num,end = '')
    #print(" " + json_data + " ", end = '')

def totxttitle(json_data):
    txtfile = open('2019_6.txt', 'a', -1, "utf-8")
    txtfile.write(json_data)
    txtfile.write(' ')
    #print(text_num,end = '')
    #print(" " + json_data + " ", end = '')

def totxttext(json_data):
    txtfile = open('2019_6.txt', 'a', -1, "utf-8")
    #txtfile.write(str(text_num) + '\t')
    txtfile.write(json_data)
    txtfile.write("\t")
    #print(json_data + " ", end = '')
    #index = input()
    #txtfile.write(index)
    txtfile.write("1\n")

##########################################

def tree():
    return defaultdict(tree)


driver = webdriver.Chrome(executable_path='chromedriver.exe')
driver.implicitly_wait(1)
driver.get('https://everytime.kr/login')

# 접속
driver.find_element_by_xpath('//*[@id="container"]/form/p[1]/input').send_keys('pcman33')  # 아이디
driver.find_element_by_xpath('//*[@id="container"]/form/p[2]/input').send_keys("p120700")  # 비밀번호
driver.find_element_by_xpath('//*[@id="container"]/form/p[3]/input').click()  # 로그인 버튼
time.sleep(2)
driver.find_element_by_xpath('//*[@id="submenu"]/div/div[2]/ul/li[1]/a').click()  # 자유게시판 클릭

everytime_link = list()  # 링크 리스트
fail_link = list()  # 실패 리스트
page_number = 8800  # 맨 처음으로 들어오는 페이지가 이미 1페이지를 긁어오기때문에 2페이지부터 넣음

#맨 처음시작할때 아니고서는 무조건 주석치기!!
txtfile = open('2019_6.txt', 'w', -1, "utf-8")
txtfile.write("text_num\ttime\ttext\tlabel\n")
txtfile.close()

for i in range(1, 620):
    driver.get('https://everytime.kr/377769/p/' + str(page_number))  # 자유 게시판 url # ex) 자유게시판의 고유 번호 = 370451
    time.sleep(1)
    html = driver.page_source  # 보고있는 페이지를 가져옴 (1페이지를 여기서 긁음)
    soup = BeautifulSoup(html, 'html.parser')

    content = soup.findAll('article')

    for url in content:
        find_url = url.find('a', attrs={'class', 'article'}).get('href')
        everytime_link.append(find_url)
    page_number = page_number - 1  # 다음 페이지로 넘어가기(한 페이지당 20개의 게시글)

with open('everytime_link.txt', 'a') as fileobject:  # 각 게시글 링크 저장
    for join_link in everytime_link:
        fileobject.write(join_link)
        fileobject.write('\n')


linkfile = open('everytime_link.txt', 'r', -1, "utf-8")
everytime_link = linkfile.readlines()
text_num=1
for url in everytime_link:

    time_now = datetime.datetime.now()  # 현재 시간 저장
    json_data = dict()

    try:
        driver.get('https://everytime.kr' + url)
        time.sleep(1)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        title = soup.find('h2', attrs={'class', 'large'}).get_text()  # 제목
        text = soup.find('p', attrs={'class', 'large'}).get_text()  # 내용
        text_time = soup.find('time', attrs={'class', 'large'}).get_text()  # 날짜
        """
        try:
            comment = soup.findAll('article')

            #for content in comment:
            #    comment_text.append(content.find('p').get_text())
            #    comment_time.append(content.find('time').get_text())
        except:
            pass  # 댓글없음
        """
        json_data['title'] = title
        json_data['text'] = text
        json_data['text_time'] = text_time

    except Exception as e:
        print(e)
        fail_link.append(url)
        continue
    encoding(json_data['text_time'])
    encoding(json_data['text'])
    encoding(json_data['title'])
    totxttime(json_data['text_time'], text_num)
    totxttitle(json_data['title'])
    totxttext(json_data['text'])
    text_num += 1

with open('./fail_url2.txt', 'w') as fileobject:
    for join_link in fail_link:
        fileobject.write(join_link)
        fileobject.write('\n')

driver.close()