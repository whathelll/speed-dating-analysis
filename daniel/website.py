import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

from itertools import combinations


def collect_info(gender, age, attractiveness, sincerity, intelligence, fun, ambitious):
	you = {}
	you['gender'] = gender; you['age'] = age; you['attr_o'] = attractiveness; 
	you['sinc_o'] = sincerity; you['intel_o'] = intelligence; you['fun_o'] = fun; 
	you['amb_o'] = ambitious
	return pd.DataFrame(you, index=[0])

def num_of_yeses(person):

	if person['gender'][0] == 0:
		yeses = 1 / (1 + np.exp(-(
			0.69339298*person['attr_o'] + 0.12862241*person['fun_o'] - 0.01857109*person['age']
			- 4.9163169)))
	else:
		yeses = 1 / (1 + np.exp(-(
			0.54519696*person['attr_o'] - 0.15007122*person['sinc_o'] + 0.2861754*person['fun_o']
			- 4.56875692)))
	return int(yeses.values[0]*100)


def admirer_plot(person, num_neighbours):
	df = pd.read_csv('Speed Dating Data.csv')

	INFO = ['iid', 'age', 'gender', 'dec_o']
	ATTS = ['attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o']
	INTERESTS = ['sports','tvsports','exercise','dining','museums','art','hiking',
	    'gaming','clubbing','reading','tv','theater','movies','concerts','music','shopping','yoga']

	model_df = df[ATTS+INFO+INTERESTS]
	model_df = model_df.dropna(how='all')
	model_df = model_df.groupby('iid').mean()
	model_df.reset_index(inplace=True)
	model_df = model_df.sort_values('dec_o', axis=0, ascending=False)

	lonely_list = model_df[model_df['dec_o'] == 0]['iid'].values

	all_reg = df[ATTS + ['iid', 'dec_o']]
	all_reg = all_reg.dropna(how='any')

	merged = all_reg.merge(model_df[INFO+ATTS+INTERESTS], on=['iid'], suffixes=('_x', ''))
	merged.drop(['attr_o_x', 'sinc_o_x', 'intel_o_x', 'fun_o_x', 'amb_o_x', 'dec_o'], axis=1, inplace=True)
	merged.rename(columns={'dec_o_x': 'dec_o'}, inplace=True)
	merged.dropna(inplace=True)

	# need this to get age in the same format (1-10) as other attributes. 
	# Can also go from 1-10 to original age if direction='backward'
	# 1 year difference is worth about .24 in the new scale
	def rescale_age(age, direction="forward"):
	    ser = df['age'] 
	    if direction.lower() == 'forward':
	        result = (10 - 1) / (ser.max() - ser.min()) * (age - ser.min()) + 1
	    elif direction.lower() == 'backward':     
	        result = (age - 1) * (ser.max() - ser.min()) / (10. - 1) + ser.min()
	    return result

	model2_df = model_df[ATTS + ['iid', 'gender', 'age']]
	# rescale age to be between 1 and 10
	model2_df['age'] = rescale_age(model2_df['age'])

	model2_df = model2_df.dropna(how='all')
	model2_df = model2_df.groupby('iid').mean()
	model2_df.reset_index(inplace=True)

	att_df = person.copy()

	# subsets model_df to a df with only the same sex as in att_series
	gender_df = model2_df[
		(model2_df['gender'] == att_df['gender'][0]) & (~model2_df['iid'].isin(lonely_list))]
    # rescale age to 1-10
	att_df['age'] = rescale_age(att_df['age'])

	sub_att_df = att_df.drop('gender', axis=1)
	difference_2Darray = gender_df[ATTS + ['age']].values - sub_att_df[ATTS + ['age']].values
        
	distance_1Darray = np.sum(np.square(difference_2Darray), axis=1)

	df2=gender_df.copy()
	df2['dist'] = distance_1Darray
	df2['age'] = rescale_age(df2['age'], 'backward')
	df2.sort_values('dist', inplace=True)

	neighbour_df = df2.head(num_neighbours)

	neighbour_df['Weight'] = (1 / neighbour_df['dist']) / ((1 / neighbour_df['dist']).sum())
	neighbour_ids = neighbour_df['iid'].values

	 # Get all dates of same gender where partner said yes
	dopple_df = df[(df['iid'].isin(neighbour_ids)) & (df['dec_o'] == 1)][['iid', 'pid', 'dec_o'] + ATTS]
	dopple_df = dopple_df.merge(neighbour_df[['iid','Weight']], on='iid')
	dopple_df = dopple_df[~dopple_df['pid'].isin(lonely_list)]

	count_df = dopple_df[['iid', 'dec_o']].groupby('iid').sum()
	count_df.reset_index(inplace=True)
	count_df.rename(columns={'dec_o': 'count'}, inplace=True)
	dopple_df = dopple_df.merge(count_df, on='iid')
	dopple_df['Weight'] = dopple_df['Weight'] / dopple_df['count']

	# get the people that said yes to the neighbours
	id_list = np.sort(np.unique(dopple_df['pid'].values))

	weight_df = dopple_df[['pid', 'Weight']]
	weight_df = weight_df.rename(columns={'pid': 'iid'})
	weight_df = weight_df.groupby('iid').sum()
	weight_df.reset_index(inplace=True)
	# Get the mean for the gender
	mean_df = df[ATTS+INTERESTS+['age']][df['gender'] != person['gender'][0]]
	mean_df['age'] = rescale_age(mean_df['age'])

	# Get the mean of the attributes/interests of all the people that said yes to ALL of our nearest neighbours
	person_df = df[ATTS+INTERESTS+['iid', 'age']][df['iid'].isin(id_list)]
	person_df = person_df.dropna(how='all')
	person_df = person_df.groupby('iid').mean()
	person_df.reset_index(inplace=True)

	person_df['age'] = rescale_age(person_df['age'])

	person_df = person_df.merge(weight_df, on='iid')
	# print person_df.head()
	person_df = person_df.multiply(person_df['Weight'], axis="index")

	# person_df[ATTS+INTERESTS+['age']].multiply(person_df['Weight'], axis="index")
	person_df.drop('Weight', axis=1, inplace=True)
	# print person_df.head()
	person_means = person_df.groupby('iid').sum()
	# Compare with the mean_df above
	difference_ser = person_means.sum() - mean_df.mean()

	# for the plot
	real_names = {'attr_o': 'Attractiveness', 'sinc_o': 'Sincerity', 'intel_o': 'Intelligence', 
		'fun_o': 'Fun', 'amb_o': 'Ambitious', 'sports': 'Sports', 'tvsports': 'TVSports', 
		'exercise': 'Exercise', 'dining': 'Dining', 'museums': 'Museums', 'art': 'Art', 
		'hiking': 'Hiking', 'gaming': 'Gaming', 'clubbing': 'Clubbing', 'reading': 'Reading', 
		'tv': 'TV', 'theater': 'Theater', 'movies': 'Movies', 'concerts': 'Concerts', 
		'music': 'Music', 'shopping': 'Shopping', 'yoga': 'Yoga', 'age': 'Age'}
	difference_ser.rename(index=real_names, inplace=True)

	x = difference_ser.index
	y = difference_ser
	fig, ax = plt.subplots()
	fig.set_size_inches(13.7, 7.5)

	sns.set(style="whitegrid", color_codes=True)
	data = difference_ser
	pal = sns.color_palette("Blues_d", len(data))
	rank = data.argsort().argsort()

	sns.barplot(x, y, ax=ax, palette=np.array(pal[::-1])[rank])

	ax.set_title('What kind of people do you attract?', fontsize=24)
	xticks = ax.set_xticklabels(labels=x, rotation=90, fontsize=20)
	ax.set_ylabel("Admirers' Attributes", fontsize=20)

	# data = titanic.groupby("deck").size()   # data underlying bar plot in question

	# pal = sns.color_palette("Greens_d", len(data))
	# rank = data.argsort().argsort()   # http://stackoverflow.com/a/6266510/1628638
	# sns.barplot(x=data.index, y=data, palette=np.array(pal[::-1])[rank])

	plt.tight_layout()
	fig.savefig('barplot.png')

""" /////////////////////////////////////////////
	/   Here is where you enter a person's age  /
	/	gender and attributes					/
	/											/
	/////////////////////////////////////////////
"""
gender = 0
age = 25
attractiveness = 8
sincerity = 9
intelligence = 8
fun = 6
ambitious = 8

# This takes the above information and gets it in the right form
you = collect_info(gender, age, attractiveness, sincerity, intelligence, fun, ambitious)
# This returns the number of yeses out of 20 that you should expect
yeses = num_of_yeses(you)
# print "yeses: ", yeses
# This returns nothing but saves 'barplot.png' to this directory
# Note. This function requires the file 'Speed Dating Data.csv' to be in this directory
admirer_plot(you, num_neighbours=3)
