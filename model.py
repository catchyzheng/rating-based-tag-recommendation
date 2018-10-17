from utils import IO
from utils import Utils as utils
from operator import itemgetter
import nltk
import pandas as pd
from nltk.corpus import words as W


class Model:
    '''
    Data Source:
        path: Datas/xxxx/
        links.csv |movieId|imdbId|tmdbId|
        movies.csv |movieId|movieTitle|genres|
        ratings.csv |userId|movieId|rating|timestamp|
        tags.csv |userId|movieId|tag|timestamp|

    Data Structure:
        tags: [[senti], [genre], [proper]]
        genreMovies: Dict: {genre: [movie]}
        sentiWords: Dict: {word: polar} global sentiment words
        defModel: Dict: {user: {movie: [tags, rating]}}
        userModel: Dict: {user: {rating: {movie}}}
        movieModel: Dict: {movie: {rating: {user}}}
    '''

    def __init__(self, oriPath, nowPath="/data/", isNew=False):
        self.oriPath = oriPath  # data source file path
        self.nowPath = nowPath  # python format data file path
        #self.LINK = oriPath + "links.csv"
        #self.MOVIE = oriPath + "movies.csv"
        #self.RATING = oriPath + "ratings.csv"
        #self.TAG = oriPath + "tags.csv"
        self.GenreMovies = nowPath + "genreMovies"
        self.SentiWords = nowPath + "sentiWords"
        self.UserModel = nowPath + "userModel"
        self.MovieModel = nowPath + "movieModel"
        self.DefModel = nowPath + "defModel"
        # self.CachedTags = nowPath + "cachedTags"
        self.GENRE = "6"
        if isNew:
            print("state: not converted")
            print("initializing...")
            self.init()
            print("converting data to python format...")
            self.convert_datas()
            print("storing data to file")
            self.store_to_file()
        else:
            print("state: converted")
            print("reading data from file...")
            self.read_from_file()
        print("done.")

    def init(self):
        self.genreMovies = dict()
        self.sentiWords = IO.read_from_file(self.SentiWords)
        if self.sentiWords is None:
            self.sentiWords = dict()
        self.userModel = dict()
        self.movieModel = dict()
        self.defModel = dict()
        # self.cachedTags = IO.read_from_file(self.CachedTags)
        # if self.cachedTags is None:
        #     self.cachedTags = dict()

    def read_from_file(self):
        self.genreMovies = IO.read_from_file(self.GenreMovies)
        self.sentiWords = IO.read_from_file(self.SentiWords)
        self.userModel = IO.read_from_file(self.UserModel)
        self.movieModel = IO.read_from_file(self.MovieModel)
        self.defModel = IO.read_from_file(self.DefModel)
        # self.cachedTags = IO.read_from_file(self.cachedTags)

    def store_to_file(self):
        IO.store_to_file(self.GenreMovies, self.genreMovies)
        IO.store_to_file(self.SentiWords, self.sentiWords)
        IO.store_to_file(self.UserModel, self.userModel)
        IO.store_to_file(self.MovieModel, self.movieModel)
        IO.store_to_file(self.DefModel, self.defModel)
        # IO.store_to_file(self.CachedTags, self.cachedTags)

    def convert_datas(self):
        self.wordset = set(W.words())
        self.notAllow = {'.', 'SYM', 'TO', 'WDT', 'WP', 'WP$',
                         'DT', 'PDT', 'CC', 'IN', 'PRP',
                         'is', 'was', 'were', 'are'}
        movieGenres = dict()

        # get datas from source
        # genres and movieGenres
        print("getting movie genres...")
        datas = pd.read_csv(self.MOVIE, dtype=object)
        for row in datas.values:
            [mId, genres] = [row[0], str(row[2])]
            if "genres" in genres:
                continue
            genres = utils.to_list(genres.lower())
            movieGenres[mId] = genres
            for genre in genres:
                self.genreMovies.setdefault(genre, []).append(mId)

        # getting datas
        print("getting datas to dict")
        print("getting tags...")
        datas = pd.read_csv(self.TAG, dtype=object)
        for row in datas.values:
            tags = self.defModel.setdefault(
                row[0], {}).setdefault(row[1], [[], [], []])
            tags = self.filter_tag(str(row[2]))
            if tags != [[], [], []]:
                allTags = self.defModel.setdefault(
                    row[0], {}).setdefault(row[1], [[], [], []])
                utils.extend_list(allTags, tags)

        print("getting ratings...")
        datas = pd.read_csv(self.RATING, dtype=object)
        # datas.rating = (datas.rating * 2).astype(int)
        for row in datas.values:
            if row[0] in self.defModel and row[1] in self.defModel[row[0]]:
                self.defModel[row[0]][row[1]].append(row[2])
        # create userModel and movieModel
        print("building user and movie models...")
        for uId, movie in self.defModel.items():
            for mId, tags in movie.items():
                if '.' not in tags[-1]:
                    tags.append("0.0")
                rating = tags[-1]
                self.userModel.setdefault(uId, {})\
                    .setdefault(rating, []).append(mId)
                self.movieModel.setdefault(mId, {})\
                    .setdefault(rating, []).append(uId)

        # add movie genres into movieModel
        for mId, genres in movieGenres.items():
            if mId in self.movieModel:
                self.movieModel[mId][self.GENRE] = genres

        # sort senti word by polar
        self.sentiWords = sorted(self.sentiWords.items(), key=itemgetter(1))

    # get word sentiment
    # if not find in sentiWords, it will cache it
    # return (positive-negative) in range [-1.0, 1.0]
    def is_sentiment(self, word):
        # return word in self.sentiWords
        if word in self.sentiWords:
            return True
        else:
            senti = utils.get_sentiment(word)
            if senti != 0:
                self.sentiWords[word] = senti
                return True
        return False

    # filter tag into three types: senti, genre, proper
    def filter_tag(self, tag):
        # remove useless words
        words = list(filter(lambda x: len(x) > 1 and x not in self.notAllow,
                            utils.to_list(tag)))
        tups = nltk.pos_tag(words)
        words = [tup[0] for tup in tups if tup[1] not in self.notAllow]
        tags = [[], [], []]
        # proper word?
        lens = len(words)
        if lens > 3:
            return tags
        elif lens == 1 and words[0].lower() not in self.wordset:
            tags[2].append(words[0])
        elif lens > 1 \
                and words[0][0].isupper() and words[1][0].isupper():
            tags[2].append(utils.to_string(words))
        else:
            # senti word or genre word
            words = utils.to_lower(words)
            for word in words:
                if word in self.genreMovies:
                    tags[1].append(word)
                elif word not in self.wordset:
                    continue
                else:
                    if self.is_sentiment(word):
                        tags[0].append(word)
        return tags
