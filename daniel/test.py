import numpy as np
import pandas as pd

def num_of_yeses(person):

	if person['gender'][0] == 0:
		yeses = 1 / (1 + np.exp(-(
			0.69339298*person['attr_o'] + 0.12862241*person['fun_o'] - 0.01857109*person['age']
			- 4.9163169)))
	else:
		yeses = 1 / (1 + np.exp(-(
			0.54519696*person['attr_o'] - 0.15007122*person['sinc_o'] + 0.2861754*person['fun_o']
			- 4.56875692)))
		print "yeses: ", yeses
	return np.round(yeses.values[0]*20, 2)


you = {}
you['gender'] = 0; you['age'] = 27; you['attr_o'] = 6; you['sinc_o'] = 6; 
you['intel_o'] = 7; you['fun_o'] = 8; you['amb_o'] = 6
you = pd.DataFrame(you, index=[0])
yeses = num_of_yeses(you)

print "you have: %s many yeses" % yeses