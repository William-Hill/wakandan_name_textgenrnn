from bs4 import BeautifulSoup
import requests


def Blackpanther_names(body):
	#parsing
	soup = BeautifulSoup(body.content, "html.parser")
	soup.prettify()
	#findall
	#return
	w = soup.find_all("p")	
	count = 0
	for c in w:
		d = c.find_all("a")
		for z in d:
			l = z.find(text = True)
			if count == 21:
				break
			else:
				print(l)
				count+=1

url = "https://marvel.fandom.com/wiki/T%27Challa_(Earth-616)"

page = requests.get(url, timeout=5)

Blackpanther_names(page)