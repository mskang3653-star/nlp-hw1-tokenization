"""HW1 Part 1 public tests. Pass these before submitting.
    <실행방법을 친절하게 알려주셨음>
    python -m pytest test_bpe_public.py -q
    or
    python test_bpe_public.py
"""

import bpe # bpe.py파일을 가지고 옴#즉 bpe.py를 수정해서 현재 이 테스트가 통과하게 만들어야 한다.

# the example on slides 15-21
TOY = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}


def test_count_words():
    got = bpe.count_words(["hug pug hug", "pun  bun", "hugs"])
    assert got == {"hug": 2, "pug": 1, "pun": 1, "bun": 1, "hugs": 1} #assert는 조건이 참이어야한다.라는 뜻으로
  #assert got == expected 이고 즉 이 해당 test_count_words()는 여러 줄의 문자열을 받아서 공백 기준으로 단어를 나누고 
  #각 단어가 몇번 나왔는지 세야하는 함수 /이건 toy가 아님

def test_merge_order():
    """The first three merges are u+g (20), u+n (16), h+ug (15)."""
    assert bpe.train_bpe(TOY, 3) == [("u", "g"), ("u", "n"), ("h", "ug")]
  #train_bpe(TOy,3) 은 bpe merge rule을 3개 학습 해라
  #즉 첫번쨰 merges는 u + g, 두 번째는 u + n, 세 번째는 h + ug 여야 한다는 뜻
  # 단어를 문자 단위로 시작하고 인접 pair 빈도를 세야하고 , 가장 자주 나오는 pair를 합쳐야 하고 이 과정을 반복하면 나온다.

def test_vocab_growth():
    """Reproduce the vocabulary growth on slides 15, 19 and 21."""
    base = ["b", "g", "h", "n", "p", "s", "u"] #알파벳 순서로정렬 한 것
    expected = {
        0: base,
        1: base + ["ug"],
        2: base + ["ug", "un"],
        3: base + ["ug", "un", "hug"],
    }
    for k, want in expected.items(): #k는 0부터 시작 , want=base
        vocab = bpe.build_vocab(TOY, bpe.train_bpe(TOY, k)) #자동으로 <sp> , <unk>이 구현 되도록 해야함
        got = [t for t in vocab if t not in bpe.SPECIALS] #got에는스페셜 토큰을 버리고 넣음 <unk>, <sp>
        assert got == want, (k, got)


def test_tokenize_matches_slide21(): #학습된 merge rule을 사용해서 실제 문장을 tokenize하느 테스트
    merges = bpe.train_bpe(TOY, 3) #[("u", "g"), ("u", "n"), ("h", "ug")]
    vocab = bpe.build_vocab(TOY, merges) #vocab만들기
    got = bpe.tokenize("hug pug pun bun hugs", merges, vocab)
    assert got == [["hug"], ["p", "ug"], ["p", "un"], ["b", "un"], ["hug", "s"]]
    # "hug pug pun bun hugs"를 토큰화 한 최종결과가 got과 같아야 한다.

def test_round_trip():
    """Round-trip holds for text within the training alphabet."""
    merges = bpe.train_bpe(TOY, 3)
    vocab = bpe.build_vocab(TOY, merges)
    for text in ["hug", "hugs bun", "pun pug hug bun hugs", "b u n"]:
        assert bpe.decode(bpe.encode(text, merges, vocab), vocab) == text
#encode와 decode가 서로 잘 맞는지 테스트 하는 과정
# bpe.encode(text,merges,vocab) 이건 token id 를 리스트로 바꾼다. 대충 이런 느낌"hugs bun" -> ["hug", "s", "<sp>", "b", "un"] -> [9, 7, 1, 2, 8]
#bpe.decode(ids, vocab),/// [9, 7, 1, 2, 8] -> "hugs bun"
#<sp>이 중요하다. ["hug", "s", "<sp>", "b", "un"] decode할때 다시 <sp> 를 " "로 바꿔야 하기 때문
def test_ids_are_in_range():
    merges = bpe.train_bpe(TOY, 3)
    vocab = bpe.build_vocab(TOY, merges)
    ids = bpe.encode("hugs and bun", merges, vocab)
    assert all(0 <= i < len(vocab) for i in ids) # all(...)은 안의 조건이 전부 참이면 True
#encode 결과로 나온 id들이 vocab범위 안에 있는지 검사
#예를 들어 and는 TOY에 없음 a, d가 따라서 <unk> 으로 처리 되야함

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS  " + name)
            except AssertionError as exc:
                fails += 1
                print("FAIL  %s  %s" % (name, exc))
            except NotImplementedError:
                fails += 1
                print("TODO  %s  (not implemented yet)" % name)
    print()
    print("%d failed" % fails)
    raise SystemExit(1 if fails else 0)
#과제를 실행 했을때 테스트를 돌려주는 테스트 러너