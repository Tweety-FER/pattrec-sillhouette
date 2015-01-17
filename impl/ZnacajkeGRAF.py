#!/usr/bin/python
import pylab as pl
import numpy as np

if __name__=="__main__":
	X=[1,2,3,4,5,6,7,8]
	K=8				#K=broj vrijednosti koje vektor smjera moze poprimiti	
	br=0;		
	with open ("chainCode.txt") as ulaz:	#citanje chain koda na ulazu
		Baza=ulaz.readlines()

	for baza in Baza:
		br+=1
		baza=baza.strip()
		brojac=[0]*8		#polje u kojem je zapisano koliko se puta pojavio koji vektor
		histogram=[0]*8		#isto ko i polje brojac al vrijednosti su normaliziranje
		Ef=0			#varijabla u kojoj se zapisuje velicina chain koda

		for j in baza:
			i=int(j)
			if i>=1 and i<=8:
				brojac[i-1]+=1	#brojanje 
				Ef+=1		#prebrojavanje sveukupne velicine chain koda jer to treba za normalizaciju


		for i in xrange(0,len(brojac)):		#normalizacija vrijednosti u brojacu i spremanje u polje histogram
			histogram[i]=(brojac[i]/float(Ef))-1/float(K)

		print "Ulazni chain ",br," je: ",baza	#ispis
		print "Izlazni F(X) ",br," je: ",np.array(histogram)
	
		pl.plot(X,histogram)	#crtanje grafa, na x osi br od 1 do 8, na y normalizirana frekvencije pojavljivanja za svaki broj

		pl.show()	#koristite ovaj pl.show da se histogram za chain code ispise na zasebnom grafu
	#pl.show()		#koristite ovaj pl.show da se svi histogrami ispisu na jednom grafu
