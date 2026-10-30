"""Part 1. Byte Pair Encoding.

Natural Language Processing (2026) -- Homework 1

Implement the algorithm on slides 14-21 using the **standard library only**.
Do not import `tokenizers`, `transformers`, or `numpy` in this file.

Specification -- grading checks that your code follows it exactly.

1. The base vocabulary is the set of characters in the training corpus.
   Use no word-boundary marker, matching the slides, which write
   "hugs" as h+u+g+s.
2. Each step merges the most frequent pair. Ties break **alphabetically**.
   Never merge a pair whose frequency is below 2.
3. `encode` splits text on whitespace, replays the merges on each word
   independently, and inserts the special token `<sp>` between words.
   `<sp>` takes no part in training; it only restores whitespace.
4. Characters absent from the training corpus encode as `<unk>`.
   **Do not raise an exception.**
5. `build_vocab` orders tokens as `<unk>`, `<sp>`, base characters in
   alphabetical order, then merged tokens in the order they were learned.
   That order defines the token ids.

Therefore `decode(encode(t)) == t` holds only for text whose whitespace is
normalized to single spaces and whose characters are in the training alphabet.

Run:
    python test_bpe_public.py
"""

from __future__ import annotations

from collections import Counter, defaultdict

    

UNK = "<unk>"
SP = "<sp>"
SPECIALS = [UNK, SP]


# -----------------------------------------------------------------------------
# 1-a. Word frequencies
# -----------------------------------------------------------------------------
def count_words(lines):
    """Count whitespace-delimited word frequencies over a sequence of lines.

    Args:
        lines: a sequence of strings.
    Returns:
        dict[str, int] mapping word to frequency.

    >>> count_words(["hug pug hug"]) == {"hug": 2, "pug": 1}
    True
    """
    # HINT: collections.Counter accepts any iterable of words.

    counts = Counter()

    for line in lines:
        words = line.split() # 공백으로 조개줌 ex) "pun bun".split()하면 ->["pun", "bun"]
        counts.update(words) #counts.update(["hug", "pug", "hug"]) 하면 내부적으로 {"hug": 2, "pug": 1}

    return dict(counts) # dict 형태로 반환




# -----------------------------------------------------------------------------
# 1-b. Merge training. **시험 출제 가능성 상당해 보임 강민성 피셜
# -----------------------------------------------------------------------------
   # 헬퍼 함수 추가
def _count_pairs(tokens):
        """한 단어 안에서 인접한 pair들의 출현 횟수를 세는 함수."""
        return Counter(zip(tokens, tokens[1:])) 

def _merge_pair(tokens,target_pair):
        """target_pair를 왼쪽 부터 겹치지 않게 합친다."""
        merged_tokens=[]
        i=0

        while i < len(tokens):
            if(i+1 <len(tokens) and (tokens[i],tokens[i+1])== target_pair):
                merged_tokens.append(tokens[i]+tokens[i+1])
                i+=2
            else:
                merged_tokens.append(tokens[i])
                i+=1
        return merged_tokens


def train_bpe(word_freqs, num_merges):
    """Learn merge rules from a word frequency table.
    단어의 빈도표를 받아서 BPE merge rule을 num_merges개까지 학습 하고 merge rule 목록을 순서대로 반환
    ex) TOY = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}. -> train_bpe(TOY, 3) -> [("u", "g"), ("u", "n"), ("h", "ug")]
    
    
    Follow the procedure on slide 14.
      1. Split each word into characters. 단어를 char로 나눈다
      2. Count adjacent pairs. 인접한 pair를 센다
      3. Select the most frequent pair (ties break alphabetically) 가장 자주 나오는걸 선택한다.
      4. Merge that pair into a single token. 각 pair를 싱글 토큰으로 합친다
      5. Repeat steps 2-4 `num_merges` times. 2번 4번 반복한다. 

    Stop early when no pair reaches a frequency of 2.

    Args:
        word_freqs: mapping from word to frequency.
        num_merges: number of merges, k.
    Returns:
        list[tuple[str, str]] of merge rules in the order they were learned.

    >>> train_bpe({"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}, 1)
    [('u', 'g')]
    """
    # HINT: rescanning every word on each merge is slow. Keep an index from each
    #       pair to the words that contain it and update only those words.
    # HINT: do not recount pair frequencies from scratch. Subtract and add the
    #       contribution of the words that changed.
    # HINT: a 5 MB corpus with 1,000 merges must finish within a few minutes.

    

    splits ={ }
    for word in word_freqs:
        splits[word]=list(word)

    #전체 말뭉치에서 각 pair가 등장한 횟수
    pair_counts = Counter() #이거면 충분하나
    #각 pair가 어떤 단어에 들어 있는지 저장
    pair_to_words =defaultdict(set)

    for word,word_freq in word_freqs.items():
        local_counts = _count_pairs(splits[word])

        for pair, local_count in local_counts.items():
             pair_counts[pair]+= local_count *word_freq
             pair_to_words[pair].add(word)

    merges = []

    for _ in range(num_merges):
         #더 이상 합칠 수 있는 pair가 없을 경우
        if not pair_counts:
              break
        #가장 높은 pair 빈도를 찾는다.
        best_count = max(pair_counts.values())

        #빈도가 2 미만인 pair는 합치지 않는다.
        if best_count <2:
             break

        #최대 빈도를 가진 pair중 알파벳순으로 가장 앞선 것을 선택한다.
        best_pair = min( pair for pair,count in pair_counts.items() if count == best_count)
        merges.append(best_pair)

        #best_pair를 가진 단어만 가져온다.
        changed_words = list(pair_to_words[best_pair])

        for word in changed_words:
            word_freq = word_freqs[word]
            old_tokens =splits[word]

             #변경전 pair 빈도를 전체 빈도에서 제거한다.
            old_counts = _count_pairs(old_tokens)

            for pair,local_count in old_counts.items():
                pair_counts[pair] -=local_count*word_freq

                if pair_counts[pair]<=0:
                    del pair_counts[pair]

                words = pair_to_words.get(pair)

                if words is not None:
                    words.discard(word)

                    if not words:
                         del pair_to_words[pair]

            new_tokens = _merge_pair(old_tokens, best_pair)
            splits[word] = new_tokens

            new_counts = _count_pairs(new_tokens)

            for pair,local_count in new_counts.items():
                pair_counts[pair] += local_count * word_freq
                pair_to_words[pair].add(word)

    return merges
            
   


# -----------------------------------------------------------------------------
# 1-c. Vocabulary
# -----------------------------------------------------------------------------
def build_vocab(word_freqs, merges):
    """Build the token list. Its index is the token id.

    Keep this order exactly:
        `<unk>`, `<sp>`, base characters (alphabetical), merged tokens
        (in the order they were learned)

    Args:
        word_freqs: mapping from word to frequency.
        merges: merge rules returned by `train_bpe`.
    Returns:
        list[str] of tokens.
    """
    # HINT: the base characters are the distinct characters of every word.
    
    base_characters =set() #학습 corpus에 존재하는 모든 문자를 모을예정 빈집합 형성 
    #/set을 사용하여 중복문자를 자동으로 제거

    for word in word_freqs:
         base_characters.update(word) # word에 있는 단어를 하나씩 꺼내서 저장

    vocab =SPECIALS.copy() #특수 토큰 부터 추가
    vocab.extend(sorted(base_characters))# 기본 문자들을 알파벳 순서로 추가
    #append를 사용하면 리스트 전체가 하나의 원소로 들어가는 일이 발생함

    for left, right in merges: #학습된 merge결과를 학습 순서대로 추가

        """ merges = [
            ("u", "g"),
            ("u", "n"),
            ("h", "ug"),
        ]"""
        merged_token = left + right
        vocab.append(merged_token)

    return vocab




# -----------------------------------------------------------------------------
# Provided. Bookkeeping tables derived from the merges and the vocabulary.
# -----------------------------------------------------------------------------
def build_tables(merges, vocab):
    """Return (ranks, alphabet, index).
     #이미 만들어진 merges와 vocab을 이후 tokenize, encode에서 편하게 조회할 수 있도록 딕셔너리와 집합으로 만드는 함수
    ranks    -- {pair: rank}, rank 0 being the merge learned first
    alphabet -- the set of base characters, the ones a word may be split into
    index    -- {token: id}
    """
    ranks = {pair: i for i, pair in enumerate(merges)} #merge 규칙에 순서번호를 붙임
    alphabet = {t for t in vocab if len(t) == 1 and t not in SPECIALS} #특수 토큰이 아니고 길이가 1인 기본 문자만 골라 집합으로 만듬
    index = {t: i for i, t in enumerate(vocab)} # 토큰을 token Id로 변환할때 사용함
    return ranks, alphabet, index

    """
    ranks:    어떤 merge를 먼저 적용해야 하는가? 랭크가 작은 거부터
    alphabet: 입력 문자가 학습 데이터에 있었는가?
    index:    이 token의 정수 ID는 무엇인가?

    tokenize -> ranks, alphabet 사용
    encode   -> index 사용
    """


# -----------------------------------------------------------------------------
# 1-d. Segmentation
# -----------------------------------------------------------------------------
def tokenize(text, merges, vocab):
    """Segment text into subwords, keeping word boundaries.

    Split on whitespace and segment each word on its own. Within a word,
    repeatedly merge the adjacent pair of the **highest rank**, which replays
    the merges in the order they were learned. A character outside the training
    alphabet becomes `<unk>`.

    Args:
        text: a string.
        merges: merge rules returned by `train_bpe`.
        vocab: token list returned by `build_vocab`.
    Returns:
        list[list[str]], one inner list per word.

    >>> merges = train_bpe({"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}, 3)
    >>> tokenize("hugs", merges, build_vocab({"hug": 10, "hugs": 5}, merges))
    [['hug', 's']]
    """
    # HINT: `build_tables` gives you the rank table and the alphabet.
    # HINT: words repeat often, so caching word -> segmentation speeds this up.
    ranks, alphabet,_ = build_tables(merges,vocab) #merge 순서와 학습된 기본 문자 집합을 가지고 옴 build_tables()이 반환 하는 세값중 2개만 사용
    cache ={} #같은 단어를 반복해서 계산하지 않기 위한 캐시/ 이미 tokenize한 단어의 결과를 저장함
    tokenized_words =[] #단어별 token 결과를 저장할 리스트
    for word in text.split():
        if word in cache:#이전에 처리한 단어라면 저장된 결과를 사용함
            tokenized_words.append(cache[word].copy())
            continue
         # 학습 corpus에 있었던 문자는 그래로 사용하고 없었던 문자는 <unk>로 바꾼다.
        tokens=[] #단어를 문자단위 token으로 분리
        for character in word:
            if character in alphabet:
                tokens.append(character)
            else:
                tokens.append(UNK)
        while True:
            candidates =[ pair for pair in zip(tokens,tokens[1:]) if pair in ranks] #현재 학습된 token들 사이의 인접 pair중
              #학습된 merge 규칙에 존재하는 pair만 찾는다.
            if not candidates: break
            best_pair = min( candidates, key=lambda pair: ranks[pair])
                #가장 먼저 학습된 merge규칙을 선택한다.
            tokens = _merge_pair(tokens,best_pair)# 선택한 pair를 현재 단어 안에서 합친다.
        cache[word]= tokens.copy() # 같은 단어가 등장 할 경우를 위해 저장함
        tokenized_words.append(tokens) #현재 단어의 tokenization 결과를 추가한다.

    return tokenized_words
# -----------------------------------------------------------------------------
# 1-e. Encoding and decoding
# -----------------------------------------------------------------------------
def encode(text, merges, vocab):
    """Encode text as token ids, inserting the id of `<sp>` between words.

    Args:
        text: a string.
        merges: merge rules returned by `train_bpe`.
        vocab: token list returned by `build_vocab`.
    Returns:
        list[int]
    """
    # HINT: segment with `tokenize`, then map each token through the index table.
    # HINT: a token missing from the vocabulary becomes the id of `<unk>`.
    _,_,index = build_tables(merges, vocab)

    tokenized_words = tokenize(text, merges, vocab)#문장을 subword단위로 나눈다.
    ids =[] #token ID를 담을 빈 리스트를 만든다.
    for word_position, tokens in enumerate(tokenized_words):
        #첫번째 단어가 아니라면 단어사이에 <sp>를 넣는다.
        if word_position > 0:
            ids.append(index[SP]) #단어와 단어 사이에 만넣음

        #현재 단어의 token들을 ID로 변환한다.
        for token in tokens: # 현재 단어를 구성하는 subword token을 하나씩 가지고온다.
            token_id = index.get(token, index[UNK]) #딕셔너리.get(찾을_key, key가_없을_때_기본값)
            ids.append(token_id)
    return ids

    """
    "hug pug"
        ↓ tokenize
    [["hug"], ["p", "ug"]]
        ↓ <sp> 삽입
    ["hug", "<sp>", "p", "ug"]
        ↓ ID 변환
    [11, 1, 6, 9]"""


def decode(ids, vocab):
    """Decode token ids back to text, rendering `<sp>` as a single space.

    Args:
        ids: a sequence of token ids.
        vocab: token list returned by `build_vocab`.
    Returns:
        str
    """
    text_parts=[] #복원할 문자열 조각을 저장할 리스트를 만든다.

    for token_id in ids : # ID를 하나 씩 가지고 온다.
        token = vocab[token_id] #아이디를 사용해 원래 토큰을 복원

        if token == SP:
            text_parts.append(" ") # 공백토큰 이면 공백
        else:
            text_parts.append(token) #아니면 진짜붙임

    return "".join(text_parts)
