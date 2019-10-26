import preprocess_text


stemmed_word = preprocess_text.prep_text()
word_count = {}
for word in stemmed_word:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
#Print in sorted order
for w in sorted(word_count, key=word_count.get, reverse=True):
    print (w, word_count[w])