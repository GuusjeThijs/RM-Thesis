import gensim
import time
import sys


def main(filename):
    start = time.time()

    # read files
    start_read = time.time()
    print('Reading corpus...')
    with open(filename, 'r') as rf:
        sentences = rf.readlines()
    print(f'Reading corpus took {time.time() - start_read} seconds\n')
    
    
    class SentenceGenerator(object):
        def __init__(self, sentences):
            self.lines = sentences
        
        def __iter__(self):
            for line in self.lines:
                yield line.split()

    generator = SentenceGenerator(sentences)

    w2v_model = gensim.models.Word2Vec(min_count=10, workers=8)
    
    print('Building vocab...')
    start_build = time.time()
    w2v_model.build_vocab(generator, progress_per=10000)
    print(f'Building vocab took {time.time() - start_build} seconds\n')

    start_train = time.time()
    print('Training word2vec model...')
    w2v_model.train(generator, total_examples=w2v_model.corpus_count, epochs=1)
    print(f'Training word2vec model took {time.time() - start_train} seconds\n')

    # Save model
    w2v_model.save('word2vec_model.model')

    print('Top 10 similar words to "nederlander" :')
    print(w2v_model.wv.most_similar(positive=["nederlander"]))

    print(f'Entire process took {time.time() - start} seconds')




if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)