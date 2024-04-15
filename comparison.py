import os
import ast
import json
import random
from wordcloud import WordCloud

def get_captions(dir):
    captions = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.txt'):
                with open(f'{root}/{file}', 'r') as f:
                    for line in f:
                        if line.startswith('Caption:'):
                            captions.append(line.split('Caption: ')[1])

    return captions

def get_tags(dir):
    tags = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.txt'):
                with open(f'{root}/{file}', 'r') as f:
                    for line in f:
                        if line.startswith('Tags:'):
                            tags.append(line.split('Tags: ')[1])

    return tags

def replace_caps(captions):
    new_captions = []
    new_words_dict = { "it" : "",
                        "top": "",
                        "view": "",
                        "down": "",
                        "up": "",
                        "a": "",
                        "the": "",
                        "on": "",
                        "sitting": "",
                        "standing": "",
                        "next": "",
                        "to": "",
                        r"'" : "",
                        "black": "",
                        "white": "",
                        "one": "",
                        "two": "",
                        "three": "",
                        "several": "",
                        }
    for caption in captions:
        for word in caption.split(' '):
            word = word.replace(r"'", '')
            if word not in new_words_dict.keys():
                new_captions.append(word)
            else:
                new_captions.append(new_words_dict[word])

    return new_captions



def replace_tags(tags):
    new_tags = []
    new_words_dict = { "1girl" : "",
                        "1boy" : "",
                        "2girls" : "",
                        "2boys" : "",
                        "solo" : "",
                        "multiple_girls" : "",
                        "no_humans" : "",
                        "still_life" : "object",
                        r"'" : "",
                        " " : ", "
    }
    for tag in tags:
        for word in tag.split(' '):
            for key in new_words_dict.keys():
                word = word.replace(key, new_words_dict[key])
            new_tags.append(word)


    return new_tags

def nomalize_data(data):
    # Total number of terms in the dataset (sum of all counts)
    total_terms = sum(data.values())

    # Calculate TF for each term (term count divided by total number of terms)
    tf_data = {term: count / total_terms for term, count in data.items()}
    return tf_data

def normalize_text(text):
    #text looks like this: "{'bookshelf': 31, 'many': 3, 'shelf': 46, 'cluttered': 3, 'kitchen': 6, 'refrigerator': 7}"
    # we want to convert it to a list of words, where the word is repeated the number of times it appears in the text
    data = ast.literal_eval(text)

    tf_data = nomalize_data(data)

    return tf_data

def generate_wordcloud(work_dir):
    captions = get_captions(work_dir)
    captions = replace_caps(captions)
    tags = get_tags(work_dir)
    tags = replace_tags(tags)

    data_pair = {}

    caption_wordcloud = WordCloud(width=1600, height=800, background_color ='white').generate(' '.join(captions))
    caption_wordcloud.to_file(f'{work_dir}/caption_wordcloud.png')
    caption_texts = WordCloud().process_text(' '.join(captions))
    caption_json = json.dumps(normalize_text(str(caption_texts)))
    data_pair['captions'] = normalize_text(str(caption_texts))
    with open(f'{work_dir}/caption_texts.json', 'w') as f:
        f.write(str(caption_json))

    tags_wordcloud = WordCloud(width=1600, height=800, background_color ='white').generate(' '.join(tags))
    tags_wordcloud.to_file(f'{work_dir}/tags_wordcloud.png')
    tags_texts = WordCloud().process_text(' '.join(tags))
    tags_json = json.dumps(normalize_text(str(tags_texts)))
    data_pair['tags'] = normalize_text(str(tags_texts))
    with open(f'{work_dir}/tags_texts.json', 'w') as f:
        f.write(str(tags_json))

    return data_pair

def unique_words(tf_data0, tf_data1):
    # Find unique words in the two datasets
    unique_words = set(tf_data0.keys()).symmetric_difference(set(tf_data1.keys()))
    unique_words_data = {}
    for word in unique_words:
        if word in tf_data0.keys():
            unique_words_data[word] = tf_data0[word]
        else:
            unique_words_data[word] = tf_data1[word]

    unique_words_data = nomalize_data(unique_words_data)
    return unique_words_data

def common_words(tf_data0, tf_data1):
    # Find common words in the two datasets
    common_words = set(tf_data0.keys()).intersection(set(tf_data1.keys()))
    common_words_data = {}
    for word in common_words:
        common_words_data[word] = tf_data0[word] + tf_data1[word]
    common_words_data = nomalize_data(common_words_data)

    return common_words_data

def pick_top_words(tf_data, n):
    # Pick the top n words from the dataset
    if n > len(tf_data):
        n = len(tf_data)
    top_words_dict = dict(sorted(tf_data.items(), key=lambda x: x[1], reverse=True)[:n])
    top_words = list(top_words_dict.keys())
    return top_words

def pick_random_words(tf_data, n):
    # Pick n random words from the dataset
    if n > len(tf_data):
        n = len(tf_data)
    random_words_dict = dict(random.sample(tf_data.items(), n))
    random_words = list(random_words_dict.keys())
    return random_words

def make_comparison(work_dir):
    data = []
    for dir in os.listdir(work_dir):
        if os.path.isdir(f'{work_dir}/{dir}'):
            data_dict = generate_wordcloud(f'{work_dir}/{dir}')
            data.append(data_dict)

    unique_captions_data = {}
    common_captions_data = data[0]['captions']

    for i in range(len(data)):
        unique_captions_data = unique_words(unique_captions_data, data[i]['captions'])
        common_captions_data = common_words(common_captions_data, data[i]['captions'])
    
    other_captions_data = unique_words(unique_captions_data, common_captions_data)

    unique_tags_data = {}
    common_tags_data = data[0]['tags']

    for i in range(len(data)):
        unique_tags_data = unique_words(unique_tags_data, data[i]['tags'])
        common_tags_data = common_words(common_tags_data, data[i]['tags'])

    other_tags_data = unique_words(unique_tags_data, common_tags_data)

    if len(common_captions_data) > 0:
        caption_wordcloud_common = WordCloud(width=1600, height=800, background_color ='white').generate_from_frequencies(common_captions_data)
        caption_wordcloud_common.to_file(f'{work_dir}/caption_wordcloud_common.png')
    if len(other_captions_data) > 0:
        caption_wordcloud_other = WordCloud(width=1600, height=800, background_color ='white').generate_from_frequencies(other_captions_data)
        caption_wordcloud_other.to_file(f'{work_dir}/caption_wordcloud_other.png')

    if len(common_tags_data) > 0:
        tags_wordcloud_common = WordCloud(width=1600, height=800, background_color ='white').generate_from_frequencies(common_tags_data)
        tags_wordcloud_common.to_file(f'{work_dir}/tags_wordcloud_common.png')
    if len(other_tags_data) > 0:
        tags_wordcloud_other = WordCloud(width=1600, height=800, background_color ='white').generate_from_frequencies(other_tags_data)
        tags_wordcloud_other.to_file(f'{work_dir}/tags_wordcloud_other.png')

    caption_for_generate = pick_top_words(common_captions_data, 15)+ pick_random_words(common_captions_data, 5) + pick_top_words(other_captions_data, 8) + pick_random_words(other_captions_data, 5)
    #remove duplicates
    caption_for_generate = list(dict.fromkeys(caption_for_generate))

    print(f"For image generation, the following words are recommended: \n{caption_for_generate}")

    return caption_for_generate
