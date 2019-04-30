#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from pyfiglet import Figlet
from prettytable import PrettyTable
import time
import sys
import warnings; warnings.simplefilter('ignore')


# In[18]:


from pyfiglet import Figlet
custom_fig = Figlet(font='slant')


# # Data Processing

# In[2]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[3]:


def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# In[4]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[5]:


start = time.time()
print('Loading file...')
#load file
md = pd.read_csv('movies_metadata.csv')
#TOP 250
#arranging the data
print('Arranging data for collecting TOP 250...')
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
#processing
C = vote_averages.mean()
m = vote_counts.quantile(0.95)
#arranging the data part 2
print('Arranging data part...')
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)
#BY GENRES
print('Processing the genres...')
s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)
#CONTENT BASED RECOMMENDER
links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
md = md.drop([19730, 29503, 35587])
#Check EDA Notebook for how and why I got these indices.
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
smd.shape
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
tfidf_matrix.shape
#cosine_similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
end = time.time()
print("Done! in {} seconds".format(end - start))
time.sleep(2)


# # recommender menu

# In[21]:


def chooseRecommendUI():
    print('input numbers for choosing movie title')
    print('s to search movie')
    print('q to quit')
    return input()


# In[20]:


def recommendUI(title,index):
    init = 0
    tits,idx = None,None
    pick = None
    custom_fig = Figlet(font='starwars')
    while init !='q':
        if init == 0:
            cls()
            print(custom_fig.renderText(title))
            print('\t\t\t\t\"{}\"\n\n'.format(md.loc[index]['tagline']))
            print(md.loc[index]['overview']+'\n')
            print('Genre: {}'.format(', '.join(md.loc[index]['genres'])))
            print('Rating: {}\n\n'.format((md.loc[index]['vote_average'])))
            tits,idx = recommender(title)
            print('people also look for...')
            printRecommend(tits)
            pick = chooseRecommendUI()
            if pick == 'q':
                init = 'q'
            elif pick.isnumeric() == True:
                init = 1
        elif init == 1:
            cls()
            print(custom_fig.renderText(tits[int(pick)]))
            print('\t\t\t\t\"{}\"\n\n'.format(md.loc[int(smd['index'][idx[int(pick)]])]['tagline']))
            print(md.loc[smd['index'][idx[int(pick)]]]['overview']+'\n')
            print('Genre: {}'.format(', '.join(md.loc[int(smd['index'][idx[int(pick)]])]['genres'])))
            print('Rating: {}\n\n'.format((md.loc[int(smd['index'][idx[int(pick)]])]['vote_average'])))
            tits,idx = recommender(tits[int(pick)])
            print('people also look for...')
            printRecommend(tits)
            pick = chooseRecommendUI()
            if pick == 'q':
                init = 'q'
            elif pick.isnumeric() == True:
                init = 1
            elif pick == 's':
                search()


# In[22]:


def recommender(title):#get list and index of recommended movie
    tits = get_recommendations(title).head(5).values.tolist()
    idx = get_recommendations(title).head(5).index.tolist()
    return tits,idx


# # search

# In[10]:


def searchTits(title):
    avg = md['title'].str.extract(r'(.*{}.*)'.format(cap(title)))
    avg =avg.drop_duplicates()
    avg =avg.dropna()
    index = avg.index.tolist()
    titsList = md.loc[avg.index.tolist()]['title'].values.tolist()
    return titsList,index


# In[11]:


def searchUI():
    cls()
    print(custom_fig.renderText('IMDB'))
    return input('Search Movie: ')


# In[12]:


def searchChoosingUI():
    print('input numbers for choosing movie title')
    print('s to search again')
    print('q to quit')
    return input()


# In[24]:


def search():
    quit = None
    while quit != 'q':
        movieSearchList,index = searchTits(searchUI())
        printRecommend(movieSearchList)
        pick = searchChoosingUI()
        if pick == 'q':
            quit = 'q'
        elif pick == 's':
            quit = None
        elif pick.isnumeric() == True:
            recommendUI(movieSearchList[int(pick)],index[int(pick)])
            quit = 'q'


# # Start menu

# In[30]:


def choose(chs):
    if chs == '1':
        cls()
        movList,category = printGenre()
        browse250(movList,category)
    elif chs == '2':
        search()

# # Menu Function

# In[16]:


def cls():
    print('\n' * 50)


# In[6]:


def printGenre():
    genreList = ['Action','TV Movie','Foreign',
                 'Comedy','Romance','Thriller',
                 'Mystery','Fantasy','Music',
                 'History','Drama','Crime',
                 'Animation','Horror','Science Fiction',
                 'War','Documentary','Western',
                 'Family','Adventure']
    x = PrettyTable()
    x.field_names = ["No","Genre"]
    for i,j in enumerate(genreList):
        #print(j)
        x.add_row([str(i),j])
    print(x)
    cho = input('choose a category: ')
    cho = int(cho)
    movList = build_chart(genreList[cho])[['title','year']].values.tolist()#get top 250 to list
    movList2 = [[str(i)]+j for i,j in enumerate(movList)]
    return movList2,genreList[cho]


# In[15]:


def printRecommend(daList):
    x = PrettyTable()
    x.field_names = ["No","Title"]
    for i,j in enumerate(daList):
        x.add_row([str(i),j])
    print(x)


# In[7]:


def cap(title):#capitalize
    title = title.lower()
    movie = title.split()
    for i,x in enumerate(movie):
        if x != 'of':
            movie[i] = movie[i].capitalize()
    return ' '.join(movie)


# In[34]:


def browse250(movList2,cat):
    quit = None
    pIndex = 0
    pages = []
    for i in range(251):
        if i % 5 == 0:
            pages.append(i)
    rPages = []
    for i,x in enumerate(pages):
        if len(pages[i:i+2]) !=1:
            rPages.append(pages[i:i+2])
    while quit != 1:
        if pIndex != 0:
            x = PrettyTable()
            x.field_names = ["No","Title", "Year"]
            for i in range(rPages[pIndex][0],rPages[pIndex][1]):
                x.add_row(movList2[i])
            cls()
            print(custom_fig.renderText('IMDB'))
            print('{} category...'.format(cat))
            print(x)
            print('input b for < \t\t\t\t\t input n for >')
            print('input numbers for choosing movie title')
            no = input()
            if no == 'n':
                pIndex = pIndex+1
            elif no == 'b':
                pIndex = pIndex-1
            elif no == 'q':
                quit = 1
            elif no.isnumeric() == True:
                #print('numeric')
                cls()
                print(custom_fig.renderText('IMDB'))
                catIndex = build_chart(cat).index.tolist()
                index = catIndex[int(no)]
                movTitle = movList2[int(no)]
                recommendUI(movTitle,index)
                quit=1
        else:
            x = PrettyTable()
            x.field_names = ["No","Title", "Year"]
            for i in range(rPages[pIndex][0],rPages[pIndex][1]):
                x.add_row(movList2[i])
            cls()
            print(custom_fig.renderText('IMDB'))
            print('{} category...'.format(cat))
            print(x)
            print('input b for < \t\t\t\t\t input n for >')
            print('input numbers for choosing movie title')
            no = input()
            if no == 'n':
                pIndex = pIndex+1
            elif no == 'b':
                print('u cannot go back..')
            elif no == 'q':
                quit = 1
            elif no.isnumeric() == True:
                #print('numeric')
                cls()
                print(custom_fig.renderText('IMDB'))
                catIndex = build_chart(cat).index.tolist()
                index = catIndex[int(no)]
                movTitle = movList2[int(no)][1]
                recommendUI(movTitle,index)
                quit = 1

# In[35]:

cls()
print('welcome to,')
print(custom_fig.renderText('IMDB'))
print('1. Genres')
print('2. Search Movie Title')
chs = input('Choose: ')
choose(chs)

