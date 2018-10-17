from model import Model
from utils import Utils as utils
from utils import IO
import time
from collections import Counter
import random
'''
    Data Structures:
        tags: [[senti], [genre], [proper]]
        genreMovies: Dict: {genre: {movie}}}
        sentiWords: Dict: {word: polar} global sentiment words
        defModel: Dict: {user: {movie: [tags, rating]}}
        userModel: Dict: {user: {rating: {movie}}}}
        movieModel: Dict: {movie: {rating: {user}}}}

    input:
    [ user: str, movie: str, rating: str ]
'''


if __name__ == '__main__':
	model = Model(oriPath="Datas/", nowPath="data/", isNew=False)
	# tests = IO.read_from_file(model.nowPath + "tests")
	tests = [('254', '260')] # ('348', '91978'),
	num_recom = 7
	num_tests = len(tests)
	weight = [0.4, 0.3, 0.3]
	g_indices = [0, 0, 0]
	T = 0
	for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
		start = time.time()
		for test in tests:
			(user, movie) = test

			
			scope = utils.get_string_range(rating, 0.5)
			# previous tags in movie/user with ratings
			movieTags = [Counter(), Counter(), Counter()]
			userTags = [Counter(), Counter(), Counter()]
			
			for rat in scope:
				if rat in model.movieModel[movie]:
					for u in model.movieModel[movie][rat]:
						if u != user:
							utils.update_counters(
								movieTags, model.defModel[u][movie])
				if rat in model.userModel[user]:
					for m in model.userModel[user][rat]:
						if m != movie:
							utils.update_counters(
								userTags, model.defModel[user][m])
			calculate proportions
			scope2 = utils.get_string_range(rating, 1.5)
			prop = [0, 0, 0]
			for rat in scope2:
			    if rat in model.userModel[user]:
			        for m in model.userModel[user][rat]:
			            for i in range(3):
			                prop[i] += len(model.defModel[user][m][i])
			sum_prop = sum(prop)
			prop = [i / sum_prop for i in prop]
			recom = list()  # recommended tags
			# add user/item high freq tags into recom
			for i in [0, 1]:
				recom.extend(userTags[i].most_common(
					int(weight[i] * num_recom)))
			for i in range(3):
				recom.extend(movieTags[i].most_common(
					int(weight[i] * num_recom)))
			# consider user preference
			if prop[1] > 0.8:  # if user like movie genre
			    recom.extend([(i, 10) for i in model.movieModel[movie][model.GENRE]])

			# consider similar movies
			if len(recom) < num_recom:
				smTags = [Counter(), Counter(), Counter()]
				similarMovie = Counter()
				if model.GENRE in model.movieModel[movie]:
					for genre in model.movieModel[movie][model.GENRE]:
						similarMovie.update(model.genreMovies[genre])
					similarMovie = utils.most_common(similarMovie, 2)
					for m in similarMovie:
						if m == movie:
							continue
						for rat in scope:
							if rat in model.movieModel[m]:
								for u in model.movieModel[m][rat]:
									utils.update_counters(
										smTags, model.defModel[u][m])
				for i in range(1):
					recom.extend(smTags[i].most_common(
						int(weight[i] * num_recom)))

			# sort recom by frequency
			recom.sort(key=lambda tup: tup[1], reverse=True)
			recom = set([t[0] for t in recom[:num_recom]])
			# from global sentiment words
			if len(recom) < num_recom:
				# model.sentiWords = list(model.sentiWords)
				lens = len(model.sentiWords)
				lef = max(int((lens / 5 * (rating))), 0)
				rig = min(lens, lef + num_recom - len(recom))
				for i in range(lef, rig):
					recom.add(model.sentiWords[i][0])
			indices = utils.get_indices(recom, real_tags)
			g_indices = [a + b for (a, b) in zip(indices, g_indices)]
		utils.print_result(recom, user, movie, rating)
		
		print("precision: ", g_indices[0])
		print("recall: ", g_indices[1])
		print("F1: ", g_indices[2])
		
			print(real_tags)
			print(recom)
		print(T)
		T = T + 1

		g_indices = [x / num_tests for x in g_indices]
		print(g_indices)
		print(num_recom)

		print(time.time() - start)


	