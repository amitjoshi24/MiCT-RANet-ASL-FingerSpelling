from bs4 import BeautifulSoup
import requests

letters = "abcdefghijklmnopqrstuvwxyz"
base_url = "https://www.signasl.org/dictionary/"

words = list()

for letter in letters:
	counter = 1
	while True:
		try:
			print ("letter: " + str(letter) + " counter: " + str(counter))
			url = base_url + letter + "/" + str(counter)
			r = requests.get(url)
			soup = BeautifulSoup(r.text, "html.parser")
			tdList = soup.find_all('td', {'class' : None})
			#print (tdList)
			if len(tdList) == 0:
				break
			for item in tdList:
				#print (item.text)
				txt = item.text

				splitted = txt.split(" ")
				for word in splitted:
					word = word.lower()
					if len(word) < 2:
						continue
					words.append(word)
			counter += 1
		except:
			break

f = open('signaslwords/signaslwords.txt', 'w')
for word in words:
	f.write(word + "\n")
f.flush()
f.close()
