import webbrowser
import requests
for line in open("E:\\1.txt"):
 url = 'https://www.baidu.com/s?wd=' + line
 headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0',}
 response = requests.get(url=url, headers=headers)
 fileName = 'a.html'
 with open(fileName, 'w', encoding='utf-8') as fp:
    fp.write(response.text)
 webbrowser.open(fileName)
