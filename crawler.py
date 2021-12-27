import requests
from bs4 import BeautifulSoup

wait_for_crawl_url = []


def crawlUrl():
    # 页面深度
    depth = 6
    # 伪装浏览器头部
    kv = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"}
    # 获得每页搜索结果
    for i in range(depth):
        url = 'https://www.liuxue86.com/gaokao/dilizhishidian/'
        if i != 0:
            url = url + str(i + 1) + "/"
        print(url)
        try:
            r = requests.get(url, headers=kv)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            html = r.text
        except:
            print("Error1")
        # 获得链接及非属性字符串
        soup = BeautifulSoup(html, 'html.parser')
        h3 = soup.find_all('li')
        for i in h3:
            a = i.a
            try:
                href = a.attrs['href']
                # 获取a标签中的文字
                if "地理" in a.text:
                    print(a.text, '\n', href)
                    wait_for_crawl_url.append(href)
            except:
                print('Error2')


crawlUrl()

from bs4 import BeautifulSoup
import re
import urllib.request, urllib.error
import xlwt
import time
import random
import re
import time
import requests
import threading
from lxml import etree
from bs4 import BeautifulSoup
from queue import Queue
from threading import Thread
import pandas as pd

findcomment = re.compile(r'<span class="short">(.*)</span>')
findtime = re.compile(r'<span class="comment-time" title="(.*)"')
findstar_list = re.compile(r'<span class="(.*)" title="(.*)"></span>')
findTitle = re.compile(r'<p class="pl2">&gt; <a href="(.*)">去 (.*) 的页面</a></p>')


def askURL(url):
    pc_agent = [
        "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0);",
        "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
        "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36"
        "Mozilla/5.0 (X11; Linux x86_64; rv:76.0) Gecko/20100101 Firefox/76.0"
    ]
    agent = random.choice(pc_agent)
    head = {'User-Agent': agent}
    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    return html


def run(q):
    while q.empty() is not True:
        datalist2 = []
        qq = q.get()
        time.sleep(0.2)
        url = str(qq)
        print(url)
        html = askURL(url)
        soup = BeautifulSoup(html, "html.parser")
        h3 = soup.find_all('p', class_="", href_="")
        for i in h3:
            if i.a != None:
                continue
            try:
                print(i.text)
                sentence = i.text
                sentence = sentence + "\n"
                f.write(sentence)
            except:
                print('Error2')
        q.task_done()


def mainn():
    global wait_for_crawl_url
    queue = Queue()
    result = []
    for s_li in wait_for_crawl_url:
        result.append(s_li)
        print(str(s_li))
    for i in result:
        queue.put(str(i))
        print(str(i))
    for i in range(10):
        thread = Thread(target=run, args=(queue,))
        thread.daemon = True
        thread.start()
    queue.join()


f = open("H:\geo\geo.txt", 'a')

mainn()
f.close()
