#!/usr/bin/python
import pylab as pl
import numpy as np

def ispisiNaStandardni(polje):
	print polje
def ispisiUDatoteku(polje,pom):
	for i in polje:
		pom.append(str(i))	#stvaranje polja u kojem su stringovi umjesto brojeva

		ispis=" ".join(pom)	#stvaranje ispisa iz polja
		
	f=open("znacajke.txt",'a+')	#ispis u datoteku
	f.write(ispis+"\n")	

if __name__=="__main__":
	K=8				#broj vrijednosti koje vektor smjera moze poprimiti
	with open ("chainCode.txt") as ulaz:	#citanje chain koda na ulazu
		Baza=ulaz.readlines()

	for baza in Baza:
		pom=[]
		baza=baza.strip()	#micanje svega osim brojeva iz baze
		brojac=[0]*8		#polje u kojem je zapisano koliko se puta pojavio koji vektor
		histogram=[0]*8		#isto kao i polje brojac, al vrijednosti su normaliziranje
		Ef=0			#varijabla u kojoj se zapisuje velicina chain koda

		for j in baza:
			i=int(j)
			if i>=1 and i<=8:
				brojac[i-1]+=1	#brojanje 
				Ef+=1		#prebrojavanje sveukupne velicine chain koda jer to treba za normalizaciju


		for i in xrange(0,len(brojac)):		#normalizacija vrijednosti u brojacu i spremanje u polje histogram
			histogram[i]=(brojac[i]/float(Ef))-1/float(K)
		
		npPolje=np.array(histogram)	#pretvaranje obicnog polja brojevau numpy polje zbog bolje ispisa

		#!!!Ispis: prvi je na standard output, a drugi u datoteku znacajke.txt. odkomontirajke koji vam triba
		ispisiNaStandardni(npPolje)
		#ispisiUDatoteku(npPolje,pom)

